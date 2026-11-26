"""The spatial march: second-order operator splitting of the one-way equation along z.

Each step of dz is the Strang splitting

    L(dz/2)  N(dz)  L(dz/2),

with L the exact linear operator (diffraction + absorption, :class:`LinearPropagator`) and N
the nonlinear one (:class:`NonlinearOperator`). The two half steps of L between consecutive
nonlinear steps are merged into one full step unless the plane in between is recorded, so a
step costs one linear step when only every few planes are recorded. Only the harmonics that
carry energy are transformed: the number of active harmonics grows as the nonlinearity
generates them.

The ``Solver`` section of the options file selects the parts and the outputs::

    "Solver": {
      "Options": {"Diff": true, "NLN": true, "Adsorp": true, "AbsorbingBC": false},
      "DiffractSolver": {"NumberOfPointsPerWaveLength": 6, "SamplingFrequencyMultiplier": 32,
                         "Type": "full"},
      "NonlinearSolver": {"method": "auto", "switch_threshold": 1e-3,
                          "amplitude_threshold": 1e-4, "cfl": 0.5},
      "ShockSmoothing": {"enabled": true, "type": "axial", "peak_multiplier": 5,
                         "post_focal_multiplier": 1, "start": 0.96, "end": 1.03,
                         "start_width": 0.01, "end_width": 0.01, "focal_length": 0.05,
                         "rise_points": 7},
      "Metrics": {"focal_quantity": "p_positive", "harmonics": 4, "levels_dB": [-6, -3, -12, -20],
                  "hydrophone_diameter": 0.0, "derating": 0.3, "face_distance": 0.003},
      "Output": {"every": 1, "plane_peaks": true, "progress_bar": true,
                 "xyz": [{"name": "p0.txt", "position": [0.02, 0.02, 0.0]}]},
      "precision": "double"
    }

All switches default to on except the absorbing layer: nonlinear propagation with absorption.

``ShockSmoothing`` adds artificial absorption against the oscillations that shocks cause. It
is on when the block is present, unless ``enabled`` is false. ``type`` chooses how:

* ``"axial"`` (the default): a fixed multiplier of the diffusivity along z, peaking around the
  focus (:class:`AxialSmoothing`; Yuldashev and Khokhlova 2011). ``start``, ``end`` and the
  widths are fractions of the focal length, which defaults to the transducer's focusing depth;
* ``"adaptive"``: at each point, the diffusivity that holds the shock there at a rise time of
  ``rise_points`` steps of the time grid (:class:`AdaptiveSmoothing`; Maxwell et al. 2017).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pywave3d._types import ComplexArray
from pywave3d.grid import Grid
from pywave3d.linear import AxialSmoothing, LinearPropagator
from pywave3d.model import WaveModel
from pywave3d.nonlinear import AdaptiveSmoothing, NonlinearMethod, NonlinearOperator
from pywave3d.options import Options
from pywave3d.recorder import PlaneRecorder, RecordSettings
from pywave3d.results import FocalQuantity, Results

log = logging.getLogger(__name__)


@dataclass
class SolverSettings:
    """Switches and numerical parameters of the march.

    Attributes:
        diffraction, nonlinearity, absorption, absorbing_layer: the parts of the model.
        paraxial: use the parabolic (KZK) diffraction operator instead of the exact one.
        method, switch_threshold, amplitude_threshold, cfl: see :class:`NonlinearOperator`.
        harmonic_tolerance: harmonics below this fraction of the fundamental's maximum are
            treated as zero (and not transformed) until the nonlinearity fills them.
        record_every: record every n-th plane (the last plane is always recorded).
        plane_peaks: also record the peak pressures over each whole plane.
        points: output points, ``(name, (x, y, z))``; recorded at the nearest grid point.
        focal_quantity, metrics_harmonics: as in :meth:`Results.metrics`.
        levels_db: levels of the focal widths, lengths and areas; the first is the main one.
        hydrophone_diameter: also report focal values averaged over a disc of this diameter
            [m], as a hydrophone measures them (0: no averaging).
        derating: derating coefficient for the exposure indices [dB / (cm MHz)].
        face_distance: distance from the source of the plane kept for the effective
            radiating area of IEC 61689 [m] (0.3 cm).
        single_precision: use complex64 fields.
        workers: threads for the FFTs.
        progress_bar: show a tqdm progress bar over the steps of the march (only when
            stderr is a terminal; otherwise progress is logged every 5 %).
        shock_smoothing: artificial absorption against shock oscillations: along z around the
            focus (:class:`AxialSmoothing`), at each shock (:class:`AdaptiveSmoothing`), or
            None (the default).
    """

    diffraction: bool = True
    nonlinearity: bool = True
    absorption: bool = True
    absorbing_layer: bool = False
    paraxial: bool = False
    method: NonlinearMethod = "auto"
    switch_threshold: float = 1e-3
    amplitude_threshold: float = 1e-4
    cfl: float = 0.5
    harmonic_tolerance: float = 1e-10
    record_every: int = 1
    plane_peaks: bool = True
    points: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)
    focal_quantity: FocalQuantity = "p_positive"
    metrics_harmonics: int = 4
    levels_db: list[float] = field(default_factory=lambda: [-6.0, -3.0, -12.0, -20.0])
    hydrophone_diameter: float = 0.0
    derating: float = 0.3
    face_distance: float = 0.003
    single_precision: bool = False
    workers: int = 1
    progress_bar: bool = True
    shock_smoothing: AxialSmoothing | AdaptiveSmoothing | None = None

    @classmethod
    def from_options(cls, opts: Options, workers: int = 1) -> SolverSettings:
        """Read the ``Solver`` node of the options (see the module docstring)."""
        d = cls()
        sw = opts.child("Options")
        absorption = sw.get("Adsorp", bool, None)
        if absorption is None:
            absorption = sw.get("Absorp", bool, d.absorption)
        diff = opts.child("DiffractSolver")
        eq = diff.get("Type", str, "full").lower()
        if eq not in ("full", "helmholtz", "parabolic", "paraxial", "kzk"):
            raise TypeError(f"options: '{diff.path}.Type' should be 'full' or 'parabolic', not {eq!r}")
        nl = opts.child("NonlinearSolver")
        met = opts.child("Metrics")
        out = opts.child("Output")
        sm = opts.child("ShockSmoothing")
        smoothing: AxialSmoothing | AdaptiveSmoothing | None = None
        kind = sm.get("type", str, "axial").lower()
        if kind not in ("axial", "adaptive"):
            raise TypeError(f"options: '{sm.path}.type' should be 'axial' or 'adaptive', not {kind!r}")
        enabled = sm.get("enabled", bool, sm.exists())
        if enabled and kind == "adaptive":
            smoothing = AdaptiveSmoothing(rise_points=sm.get("rise_points", float, AdaptiveSmoothing().rise_points))
        elif enabled:
            ds = AxialSmoothing()
            smoothing = AxialSmoothing(
                focal_length=sm.get("focal_length", float, ds.focal_length),
                peak_multiplier=sm.get("peak_multiplier", float, ds.peak_multiplier),
                post_focal_multiplier=sm.get("post_focal_multiplier", float, ds.post_focal_multiplier),
                start=sm.get("start", float, ds.start),
                end=sm.get("end", float, ds.end),
                start_width=sm.get("start_width", float, ds.start_width),
                end_width=sm.get("end_width", float, ds.end_width),
            )
        levels = met.get("levels_dB", list[float], d.levels_db)
        points: list[tuple[str, tuple[float, float, float]]] = []
        for p in out.children("xyz"):
            pos = p.get("position", list[float], None)
            if pos is None or len(pos) != 3:
                raise TypeError(f"options: '{p.path}.position' should be three numbers (x, y, z)")
            points.append((p.get("name", str, f"point{len(points)}.txt"), (pos[0], pos[1], pos[2])))
        return cls(
            diffraction=sw.get("Diff", bool, d.diffraction),
            nonlinearity=sw.get("NLN", bool, d.nonlinearity),
            absorption=absorption,
            absorbing_layer=sw.get("AbsorbingBC", bool, d.absorbing_layer),
            paraxial=eq in ("parabolic", "paraxial", "kzk"),
            method=nl.get("method", str, d.method).lower(),  # type: ignore[arg-type]
            switch_threshold=nl.get("switch_threshold", float, d.switch_threshold),
            amplitude_threshold=nl.get("amplitude_threshold", float, d.amplitude_threshold),
            cfl=nl.get("cfl", float, d.cfl),
            record_every=max(1, out.get("every", int, d.record_every)),
            plane_peaks=out.get("plane_peaks", bool, d.plane_peaks),
            progress_bar=out.get("progress_bar", bool, d.progress_bar),
            points=points,
            focal_quantity=met.get("focal_quantity", str, d.focal_quantity),  # type: ignore[arg-type]
            metrics_harmonics=met.get("harmonics", int, d.metrics_harmonics),
            levels_db=levels,
            hydrophone_diameter=met.get("hydrophone_diameter", float, d.hydrophone_diameter),
            derating=met.get("derating", float, d.derating),
            face_distance=met.get("face_distance", float, d.face_distance),
            single_precision=opts.get("precision", str, "double").lower() == "single",
            workers=workers,
            shock_smoothing=smoothing,
        )


class WaveSolver:
    """Marches the source-plane field of a model through the grid.

    Args:
        model: the model (medium, source, transducer).
        grid: the grid; by default built by ``model.make_grid()``.
        settings: the solver settings; by default nonlinear propagation with absorption.

    Shock smoothing without a focal length takes the transducer's focusing depth; it is an
    error if that is 0 (an unfocused piston).
    """

    def __init__(self, model: WaveModel, grid: Grid | None = None, settings: SolverSettings | None = None) -> None:
        self.model = model
        self.grid = grid if grid is not None else model.make_grid()
        self.settings = settings if settings is not None else SolverSettings()
        s = self.settings
        self.dtype = np.complex64 if s.single_precision else np.complex128
        smoothing = s.shock_smoothing if isinstance(s.shock_smoothing, AxialSmoothing) else None
        adaptive = s.shock_smoothing if isinstance(s.shock_smoothing, AdaptiveSmoothing) else None
        if smoothing is not None and smoothing.focal_length == 0:
            depth = model.transducer.focusing_depth
            if depth <= 0:
                raise ValueError("shock smoothing: set focal_length (the transducer is unfocused)")
            smoothing = dataclasses.replace(smoothing, focal_length=depth)
        self.smoothing = smoothing
        self.adaptive = adaptive
        self.linear = LinearPropagator(
            self.grid,
            model.medium,
            model.frequency,
            diffraction=s.diffraction,
            absorption=s.absorption,
            paraxial=s.paraxial,
            layer=model.layer if s.absorbing_layer else None,
            dtype=self.dtype,
            workers=s.workers,
            smoothing=smoothing,
        )
        self.nonlinear = (
            NonlinearOperator(
                model.medium,
                model.frequency,
                self.grid.n_harmonics,
                method=s.method,
                switch_threshold=s.switch_threshold,
                amplitude_threshold=s.amplitude_threshold,
                cfl=s.cfl,
                workers=s.workers,
                smoothing=adaptive,
            )
            if s.nonlinearity and model.medium.nonlinearity > 0
            else None
        )
        self.field: ComplexArray = model.initial_field(self.grid, self.dtype)
        self.n_active = self._active_harmonics()

    def _active_harmonics(self) -> int:
        """Number of leading harmonics with energy (the rest are zero)."""
        amp = np.abs(self.field).reshape(self.field.shape[0], -1).max(axis=1)
        top = amp.max()
        if top == 0:
            return 1
        nz = np.nonzero(amp > self.settings.harmonic_tolerance * top)[0]
        return int(nz[-1]) + 1

    # -- the march -----------------------------------------------------------------------

    def record_settings(self) -> RecordSettings:
        """What the run records (the recording part of the solver settings)."""
        s = self.settings
        return RecordSettings(
            focal_quantity=s.focal_quantity,
            plane_peaks=s.plane_peaks,
            points=list(s.points),
            metrics_harmonics=s.metrics_harmonics,
            levels_db=list(s.levels_db),
            hydrophone_diameter=s.hydrophone_diameter,
            derating=s.derating,
            face_distance=s.face_distance,
            workers=s.workers,
        )

    def run(self) -> Results:
        """March from z0 to z0 + lz and return what was recorded."""
        g, s = self.grid, self.settings
        k = g.n_harmonics
        nsteps = g.n_steps
        lx, ly, lz = g.lengths
        points_in = []
        for name, pos in s.points:
            x, y, zp = pos
            inside = (
                g.x0 - g.dx / 2 <= x <= g.x0 + lx + g.dx / 2
                and g.y0 - g.dy / 2 <= y <= g.y0 + ly + g.dy / 2
                and g.z0 - g.dz / 2 <= zp <= g.z0 + lz + g.dz / 2
            )
            if inside:
                points_in.append((name, pos))
            else:
                log.warning("output point %s at %s is outside the domain: not written", name, pos)
        # record every n-th plane, and always the planes of the output points, the source,
        # the last plane and the face plane (IEC 61689)
        point_planes = {g.nearest(*pos)[2] for _, pos in points_in}
        face_step = g.nearest(*g.centre, g.z0 + s.face_distance)[2]
        forced = point_planes | {0, nsteps, face_step}
        record = [i in forced or i % s.record_every == 0 for i in range(nsteps + 1)]
        settings = self.record_settings()
        settings.points = points_in
        medium = self.model.medium
        recorder = PlaneRecorder(
            g,
            k,
            frequency=self.model.frequency,
            sound_speed=medium.sound_speed,
            impedance=medium.impedance,
            settings=settings,
        )
        timings = {"linear": 0.0, "nonlinear": 0.0, "record": 0.0}
        godunov_from = math.nan

        def store(step: int) -> None:
            t0 = time.perf_counter()
            recorder.add(float(g.z[step]), self.field, self.n_active)
            timings["record"] += time.perf_counter() - t0

        log.info("Grid: %s", g.summary())
        log.info(
            "Model: diffraction %s, nonlinearity %s, absorption %s, absorbing layer %s",
            *(
                "on" if v else "off"
                for v in (s.diffraction, self.nonlinear is not None, s.absorption, s.absorbing_layer)
            ),
        )
        if self.smoothing is not None:
            sm = self.smoothing
            f = sm.focal_length
            log.info(
                "Shock smoothing: diffusivity x %g from z = %.4e m, x %g from z = %.4e m (widths %.2e, %.2e m)",
                sm.peak_multiplier,
                g.z0 + sm.start * f,
                sm.post_focal_multiplier,
                g.z0 + sm.end * f,
                sm.start_width * f,
                sm.end_width * f,
            )
        if self.adaptive is not None:
            if self.nonlinear is None:
                log.warning("Adaptive shock smoothing needs the nonlinear step: it does nothing here")
            else:
                log.info(
                    "Adaptive shock smoothing: shock rise time %.3e s (%g steps of the time grid)",
                    self.nonlinear.rise_time,
                    self.adaptive.rise_points,
                )
        max_artificial = 0.0
        store(0)
        record_before = timings["record"]
        pending = False  # a half linear step is owed from the previous step
        progress = max(1, nsteps // 20)
        t_start = time.perf_counter()
        bar = tqdm(
            total=nsteps,
            desc="march",
            unit="step",
            disable=None if s.progress_bar else True,  # None: off when stderr is not a terminal
            dynamic_ncols=True,
        )
        # route log records through tqdm.write so they print above the bar
        with bar, logging_redirect_tqdm() if not bar.disable else contextlib.nullcontext():
            for step in range(1, nsteps + 1):
                t0 = time.perf_counter()
                z_prev = float(g.z[step - 1])
                if self.nonlinear is None:
                    self.linear.apply(self.field, 1.0, self.n_active, z_prev)
                    t1 = time.perf_counter()
                else:
                    # the owed half step of the previous step starts dz/2 before z_prev
                    if pending:
                        self.linear.apply(self.field, 1.0, self.n_active, z_prev - g.dz / 2)
                    else:
                        self.linear.apply(self.field, 0.5, self.n_active, z_prev)
                    t1 = time.perf_counter()
                    method = self.nonlinear.apply(self.field, g.dz, self.n_active)
                    if method == "godunov" and math.isnan(godunov_from):
                        godunov_from = float(g.z[step - 1] + g.dz / 2)
                        log.info("Shock-capturing (Godunov) nonlinear step from z = %.4e m", godunov_from)
                    max_artificial = max(max_artificial, self.nonlinear.last_artificial_diffusivity)
                    self.n_active = self._active_harmonics()
                    pending = True
                t2 = time.perf_counter() if self.nonlinear is not None else t1
                if record[step]:
                    if pending:
                        self.linear.apply(self.field, 0.5, self.n_active, float(g.z[step]) - g.dz / 2)
                        pending = False
                    store(step)
                t3 = time.perf_counter()
                timings["linear"] += (t1 - t0) + (t3 - t2)
                timings["nonlinear"] += t2 - t1
                bar.update()
                if not bar.disable:
                    bar.set_postfix(
                        z=f"{g.z[step]:.3e} m",
                        p_axis=f"{recorder.axis_p_plus_max:.3e} Pa",
                        harmonics=self.n_active,
                        refresh=False,
                    )
                elif step % progress == 0 or step == nsteps:
                    log.info(
                        "step %*d / %d  z = %.4e m  max p+ on axis %.4e Pa  harmonics %d",
                        len(str(nsteps)),
                        step,
                        nsteps,
                        g.z[step],
                        recorder.axis_p_plus_max,
                        self.n_active,
                    )
        if self.nonlinear is not None:
            self.nonlinear.close()
        if self.adaptive is not None and self.nonlinear is not None:
            log.info("Adaptive shock smoothing: largest artificial diffusivity %.3e m^2/s", max_artificial)
        timings["linear"] -= timings["record"] - record_before
        timings["total"] = time.perf_counter() - t_start
        log.info(
            "Done in %.2f s (linear %.2f s, nonlinear %.2f s, recording %.2f s)",
            timings["total"],
            timings["linear"],
            timings["nonlinear"],
            timings["record"],
        )
        return recorder.finish(
            alpha=medium.alpha(self.model.frequency, np.arange(1.0, k + 1)) if s.absorption else np.zeros(k),
            focusing_depth=self.model.transducer.focusing_depth if self.model.source is None else 0.0,
            godunov_from=godunov_from,
            timings=timings,
        )


def simulate(
    options: str | Path | Options | None = None, workers: int = 1, progress_bar: bool | None = None
) -> Results:
    """Run a simulation from an options file (or node), or the water default if None.

    Args:
        options: path of a JSON options file, an :class:`Options` node, or None for the
            defaults (a 50 W, 1 MHz focused transducer in water, nonlinear and lossy).
        workers: threads for the FFTs.
        progress_bar: show the progress bar of the march; None takes it from the options
            (``Solver.Output.progress_bar``, default true).
    """
    if options is None:
        opts = Options({})
    elif isinstance(options, Options):
        opts = options
    else:
        opts = Options.load(options)
    model = WaveModel.from_options(opts.child("Model"))
    solver_opts = opts.child("Solver")
    grid = model.make_grid(solver_opts.child("DiffractSolver"))
    settings = SolverSettings.from_options(solver_opts, workers)
    if progress_bar is not None:
        settings.progress_bar = progress_bar
    return WaveSolver(model, grid, settings).run()

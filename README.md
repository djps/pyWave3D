# pyWave3D

[![CI](https://github.com/djps/pyWave3D/actions/workflows/ci.yml/badge.svg)](https://github.com/djps/pyWave3D/actions/workflows/ci.yml)

Pseudo-spectral spatial-marching scheme for nonlinear ultrasound from holography data.

pyWave3D computes nonlinear ultrasound beams in three dimensions. It reads JSON options files and source-plane inputs (for example a hologram measured on a plane in front of the transducer).

By default, pyWave3D models **nonlinear propagation in water with thermoviscous absorption**, which grows with the square of the frequency.

## The method

The pressure on each plane $z = \text{const}$ is stored as the complex values of its harmonics in the retarded time $\tau = t - z/c_0$:

```math
p(x, y, z, \tau) = \mathfrak{Re} \sum_{n=1}^{K} P_n(x, y, z) \, e^{+i n \omega \tau} .
```

From the real and the imaginary parts, the phase and the amplitude can be computed.

The field is marched from the source plane along $z$ by second-order (Strang) operator splitting of the one-way Westervelt equation. This is the approach of the Moscow State University HIFU simulators (Yuldashev and Khokhlova, *Acoust. Phys.* 57, 2011; Yuldashev et al., *IEEE TUFFC* 68, 2021). Each step of length $\Delta z$ is $L(\Delta z/2) \, N(\Delta z) \, L(\Delta z/2)$.

### Linear step $L$

The linear step (`pywave3d.linear`) applies diffraction, absorption and dispersion together, exactly, for each harmonic in the angular spectrum domain. Each plane-wave component $(k_x, k_y)$ of harmonic $n$ is multiplied by

```math
\exp\!\left[ -i \left( k_{z,n} - k_n \right) \Delta z - \left( \alpha_n + i \, \Delta k_n \right) \Delta z \right], \qquad k_{z,n} = \sqrt{k_n^2 - k_x^2 - k_y^2} .
```

Evanescent components decay. With `"Type": "parabolic"` the KZK (paraxial) operator is used instead.

### Nonlinear step $N$

The nonlinear step (`pywave3d.nonlinear`) integrates

```math
\frac{\partial p}{\partial z} = \frac{\beta}{\rho_0 c_0^3} \, p \, \frac{\partial p}{\partial \tau}
```

at each transverse point, in one of two ways. While the spectrum is narrow, it is integrated in the frequency domain by 4th-order Runge–Kutta, with the quadratic term evaluated pseudo-spectrally and free of aliasing (3/2 rule). Once shocks form, it is integrated in the time domain by a conservative MUSCL–Godunov scheme that captures the shocks; truncating to $K$ harmonics then removes the energy a shock dissipates.

`"method": "auto"` (the default) switches from the first to the second once the $K$-th harmonic exceeds `switch_threshold` ($10^{-3}$) times the fundamental.

### Absorption

`thermoviscous` (the default) gives $\alpha = \delta \omega^2 / (2 c_0^3)$, from the sound diffusivity $\delta$. For water $\delta = 4.33 \times 10^{-6}$ m²/s, giving 0.026 Np/m at 1 MHz.

`power_law` gives $\alpha_n = \alpha_0 n^y$, optionally with the Kramers–Kronig dispersion that the power law requires.

### Performance

The transverse sizes $n_x$ and $n_y$ are rounded up to FFT-friendly sizes, and only the harmonics that hold energy are transformed. The FFTs are multithreaded (`scipy.fft`, `workers`).

The two linear half-steps between unrecorded planes are merged into one. This is exact without an absorbing layer; a layer does not commute with diffraction, so with one, `Output.every` changes results at the $10^{-3}$ level.

Points below $10^{-4}$ of the plane maximum skip the nonlinear step, and the nonlinear step works on cache-sized chunks of points on a thread pool.

Plane peaks are computed only at points where the bound $\sum_n \lvert P_n \rvert$ can beat the values already known.

### Defaults

The defaults are water at 20 °C ($c_0 = 1482$ m/s, $\rho_0 = 998$ kg/m³, $\beta = 3.5$, $\delta = 4.33 \times 10^{-6}$ m²/s) and a 1 MHz, 50 W focused piston (radius 12.5 mm, focus 50 mm), on a 40 × 40 × 100 mm domain with 6 points per wavelength and 16 harmonics.

## Installing

Requires Python ≥ 3.10, numpy, scipy and beartype. From the repository root:

```
pip install -e ".[test]"   # or ".[dev]" for ruff and mypy as well
```

## Running

```
pywave3d [nthreads] [options.json] [-o OUTDIR] [-q | -v]
python -m pywave3d 8 examples/water_focused.json -o out
```

With no options file, pyWave3D runs the default water case. It has 405 steps on a 168 × 168 grid and takes about 45 s on 8 threads.

From Python:

```python
import pywave3d as pw

res = pw.simulate("examples/water_focused.json", workers=8)  # or pw.simulate() for the defaults
m = res.metrics()
print(m["focus"])  # position, p+, p-, harmonics, intensity, heating rate, MI
res.save("out")
```

Or build the pieces yourself:

```python
model = pw.WaveModel(
    medium=pw.Medium.water(), transducer=pw.Transducer(outer_radius=0.01, focusing_depth=0.05, power=30.0)
)
grid = pw.Grid.from_lengths(0.04, 0.04, 0.08, wavelength=model.wavelength, nt=32)
res = pw.WaveSolver(model, grid, pw.SolverSettings(method="auto", workers=8)).run()
```

To run the reference case, `options/woptions.json` with its source `data/pxy_at_z0.txt` (a uniform 1 MPa, 9.6 mm square piston at 1 MHz):

```
python examples/woptions_example.py --out woptions_out --workers 8   # about a minute
```

It runs the file as given: linear, on a 10 mm periodic domain that the piston fills, so the field is close to a plane wave. Then it runs the same source in open water: a 60 mm domain with a PML, 100 mm long, nonlinear with 16 harmonics. Each run's outputs go to `woptions_out/<run>/`, with a summary and, if matplotlib is installed, a figure (`woptions.png`) of the axis, the harmonics, the $xz$ plane and the waveforms at the output points.

## Outputs of other solvers

`Results` and its metrics, IEC parameters and files do not depend on the solver. They are built by `PlaneRecorder` (`recorder.py`) from transverse planes holding the harmonics of a periodic field. The march feeds it directly. The modules of `pywave3d.preprocess` reduce the output of other solvers to such planes:

| module | solvers | what it does |
| --- | --- | --- |
| `preprocess.spectral` | frequency-domain pseudo-spectral: marching, angular-spectrum, KZK (e.g. k-Wave's `angular_spectrum_cw`) | reorders the axes (`layout`, e.g. `"zkxy"` or `"xyz"` for one harmonic), conjugates an $e^{-i \omega t}$ convention (`time_sign=-1`), turns absolute phases into retarded ones (`phase="absolute"`) |
| `preprocess.time_domain` | time-domain pseudo-spectral (k-space), e.g. k-wave-python | fits the harmonics by least squares over the last `periods` periods of the record (exact for any time step; warns if more than 1% of the signal is left over, i.e. not yet steady), turns the `propagation` axis into $z$. `from_kwave` reads `kspaceFirstOrder`'s `(n_sensor, Nt)` data in C order of a block sensor mask, with `record_start_index` and the `kgrid` coordinates |
| `preprocess.fdtd` | finite-difference time-domain, e.g. fullwave25 | as `time_domain`, and by default also fits a constant and a linear drift (left by absorbing layers). `from_fullwave` places the sensors by their grid indices (`sensor.outcoords`, any order), with the time decimation `sampling_modulus_time` and fullwave25's $x$ as the axial direction. It reads only attributes, so fullwave25 need not be installed |

```python
from pywave3d.preprocess import time_domain

data = kspaceFirstOrder(kgrid, medium, source, sensor)  # a CW source, sensor = a block
res = time_domain.from_kwave(
    data, kgrid, frequency=1e6, sound_speed=1500, density=1000, sensor_mask=sensor.mask, propagation="x", n_harmonics=4
)
res.metrics()["iec"], res.save("out")
```

All of them take the recording settings (`RecordSettings`: output points, focal quantity, levels, hydrophone diameter, derating), the beam axis (`axis_xy`, default the plane centre), and optionally the absorption per harmonic (`alpha`, for the heating rate). A 2D field (one transverse axis) gives $n_y = 1$ and $\Delta y = 1$ m: its power and areas are per metre and its $y$ widths are NaN. The time-domain reduction assumes a steady continuous wave over the window: pulsed fields are not covered.

## Outputs

The run writes these files to the output directory:

| file | contents |
| --- | --- |
| `axial.txt` | the **beam axis** (the plane centre, or `axis_xy` for other solvers): $z$, $p_+$, $p_-$, intensity, heating rate, pulse-intensity integral, derated $p_r$ and $I_{\mathrm{ta}}$, $\lvert P_1 \rvert \ldots \lvert P_4 \rvert$ |
| `planes.txt` | every plane: $z$, power, max $p_+$, min $p_-$, max intensity, beam centre $(x, y)$ |
| `focal_plane.txt` | the **focal plane**, the plane of the maximum of the focal quantity on the axis: $x$, $y$, $p_+$, $p_-$, intensity, $\lvert P_1 \rvert \ldots \lvert P_4 \rvert$ |
| `metrics.json` | the **focus** ($x$, $y$, $z$, $p_+$, $p_-$, harmonics, intensity, heating rate), the spatial peaks of $p_+$, $p_-$ and intensity with their positions, the focal length, widths and areas at each of `levels_dB`, focal gain and shift, power, and where the shock-capturing scheme took over. Under `iec`: every IEC field parameter the scheme can give (see below) |
| `results.npz` | every recorded array: the axis harmonics, the $p_+$/$p_-$ and $\lvert P_1 \rvert$ profiles along $x$ and $y$ through the axis on every plane, the plane extremes, the power through every plane, the running maximum of $p_+$, the focal plane, the plane of the spatial peak and the last plane |
| `<name>` | the waveform at each point listed in `Solver.Output.xyz`: $t$, $p$, instantaneous intensity |

The focal quantity is set by `Solver.Metrics.focal_quantity`: `p_positive` (the default), `p_negative` or `first_harmonic`.

### IEC field parameters

[docs/iec_outputs.md](docs/iec_outputs.md) lists the field parameters that the IEC standards ask for (IEC 62127-1, 61828, TS 62556, 62555, 62359, 61689). For each, it says how to compute it in this scheme and where pyWave3D records it. pyWave3D records all of them in every run (`pywave3d.iec`, the `iec` section of `metrics.json`):

| group | parameters |
| --- | --- |
| frequency | the acoustic working frequency |
| pressure | $p_c$ and $p_r$, and hydrophone-averaged values |
| intensity | $I_{\mathrm{ta}}$ (plane-wave and exact), $I_{\mathrm{spta}}$, $I_{\mathrm{sppa}}$, ppsi and pii |
| power | plane-wave and exact, and through the focal area |
| beam geometry | the beam-alignment axis, focal widths, depths and areas at every level, and side lobes |
| safety indices | derated MI and the thermal indices TIS, TIB and TIC |
| areas | $A_{\mathrm{BCS}}$, $A_{\mathrm{ER}}$, the effective intensity and $R_{\mathrm{BN}}$ |
| nonlinearity | steepness and shock parameters |

The exceptions are pulse duration and the scanned-mode indices, which a single periodic beam cannot represent. A test checks that every key the document names is produced.

## Options

Keys are case-insensitive, dotted paths walk nested objects, and a value of the wrong type is an error that names the key. Every key is optional. [examples/water_focused.json](examples/water_focused.json) spells out the defaults.

| section | keys (defaults) |
| --- | --- |
| `Model.Geometry` | `lx`, `ly`, `lz` (0.04, 0.04, 0.1 m), `origin` ([0, 0, 0]) |
| `Model.PhysicalProperties` | `BaseFrequency` (1e6), `SpeedofSound` (1482), `Density` (998), `NonlinearityCoefficient` (3.5), `Diffusivity` (4.33e-6) |
| `Model.PhysicalProperties.Attenuation` | `type`: `thermoviscous` (default) or `power_law`; `pre_factor` ($\alpha_0$ in Np/m at the base frequency); `power`; `dispersion` (false) |
| `Model.Transducer` | `outer_radius` (0.0125), `inner_radius` (0), `focusing_depth` (0.05), `power` (50 W), or `amplitude` (Pa) |
| `Model.Initial` | `InputFile`: a text file of `x y Re(p) Im(p)` lines, or `"type": "binary"` with `InputFile.{xcord, ycord, real, imag}` (raw coordinate files and one raw real and one raw imaginary file per harmonic, column-major). Without it, the source is the transducer. Relative names are looked up next to the options file, in `../data` relative to it, and in the current directory |
| `Model.AbsorbingBC` | `type` (`sponge` or `pml`), `width` (m), `power` (2), and `strength` ($\sigma$ at the edge, Np/m) or `reflection` (round-trip attenuation $R$ at normal incidence, from which $\sigma$ is set: $\sigma = (m+1) \ln(1/R) / (2w)$, with $m$ the `power` and $w$ the `width`); for the PML, `levels` (4). A layer inside the lateral edges, on when `Solver.Options.AbsorbingBC` is true |
| `Solver.Options` | `Diff`, `NLN`, `Adsorp` (all true), `AbsorbingBC` (false) |
| `Solver.DiffractSolver` | `NumberOfPointsPerWaveLength` (6, or [x, y, z]), `SamplingFrequencyMultiplier` (32 samples per period, i.e. 16 harmonics), `NumberOfPoints` ([nx, ny, nz]), `Type` (`full` or `parabolic`), `FastFFTSizes` (true: round $n_x$ and $n_y$ up to FFT-friendly even sizes) |
| `Solver.NonlinearSolver` | `method` (`auto`, `spectral`, `godunov`), `switch_threshold` (1e-3), `amplitude_threshold` (1e-4), `cfl` (0.5) |
| `Solver.Metrics` | `focal_quantity` (`p_positive`), `harmonics` (4), `levels_dB` ([-6, -3, -12, -20]; the first is the main level), `hydrophone_diameter` (0 m: no averaging), `derating` (0.3 dB/(cm MHz)), `face_distance` (0.003 m, the plane for $A_{\mathrm{ER}}$) |
| `Solver.Output` | `every` (record every $n$-th plane, 1), `plane_peaks` (true), `xyz` (points `{"name", "position"}`) |
| `Solver.precision` | `double` (default) or `single` |

The reference file `options/woptions.json` sets `"NLN": false`, so it runs linear. A missing switch defaults to on.

## Conventions

### Phase

Phases follow $p = \mathfrak{Re} \sum_n P_n e^{+i n \omega t}$. A source converging on $z = d$ has the phase $e^{+i k \left( \sqrt{r^2 + d^2} - d \right)}$.

### Grid

The grid includes both ends of the domain, so $\Delta x = l_x / (n_x - 1)$. The sizes $n_x$ and $n_y$ are even (and FFT-friendly unless `FastFFTSizes` is false, or they are given explicitly), and the beam axis is the grid point $(n_x/2, n_y/2)$.

### Absorbing layers

The angular spectrum method treats the transverse plane as periodic. Keep the field well inside the domain, or enable an absorbing layer.

#### Sponge

`sponge` multiplies every component by $e^{-\sigma \Delta z}$ per step, whatever its direction.

#### Perfectly matched layer

`pml` is a perfectly matched layer, the complex coordinate stretch $\partial_x \to \left( 1 - i \sigma / k_n \right)^{-1} \partial_x$. A plane wave $(k_x, k_y)$ decays per unit $z$ by

```math
\frac{\sigma_x k_x^2}{k_n k_z} + \frac{\sigma_y k_y^2}{k_n k_z} ,
```

so waves leaving through the side are absorbed and waves running along $z$ next to the edge are left alone. Since $\sigma$ varies in $x$ and $y$ while the decay depends on $(k_x, k_y)$, the symbol is applied at `levels` values of $\sigma$ and interpolated pointwise (phase shift plus interpolation). Each level costs one inverse FFT per direction and harmonic. The PML needs diffraction on.

#### Comparison

On a Gaussian beam leaving the side at 25°, compared with a domain 5× wider, the PML leaves 0.43× the sponge's error. Along $z$ next to the edge it leaves 0.05× the sponge's error, because the sponge eats into the beam. At 45–60° the two are alike.

## Type checking

`import pywave3d` installs [beartype](https://github.com/beartype/beartype) on the whole package, so every annotated argument and return value is checked at run time. Array annotations check the dtype kind too: a real array where a complex field is expected is rejected. Ints are accepted where floats are expected. The checks cost microseconds per call. To turn them off, set `PYWAVE3D_NO_BEARTYPE=1` before importing.

Two things keep this complete: ruff's `ANN` rules require every function in `src/` to be annotated, and `tests/test_beartype.py` checks that beartype wraps every annotated function in every module, and fails if `PYWAVE3D_NO_BEARTYPE` is set. So the test suite always runs with the checks on.

## Development

```
pip install -e ".[dev]"
pre-commit install          # runs ruff check, ruff format and the beartype tests before each commit
ruff check . && ruff format --check .
```

Ruff settings are in `pyproject.toml` (line length 120). GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check` and the full test suite on Python 3.10, 3.12 and 3.13 for every push to `main` and every pull request.

## Tests

```
pytest                      # about 5 s
pytest tests/test_kwave.py  # needs k-wave-python
```

### `test_linear.py`

Checks one step against its closed form: the phase of a transverse mode, evanescent decay, absorption, the paraxial operator, half steps, the sponge, and the PML: a plane wave untouched, the decay of a transverse mode at the edge (exact and paraxial), and wrap-around of a tilted and an edge-hugging beam against a 5× wider domain.

### `test_nonlinear.py`

Checks the nonlinear step against the exact plane-wave solutions: Fubini before the shock (both integrators), and the sawtooth decay $2 p_0 / \left( n (1 + \sigma) \right)$ after it (Godunov). It also checks energy conservation, second-harmonic growth, the automatic switch, and that results do not depend on the number of threads.

### `test_solver.py`

Runs end to end: the linear focused field on the axis against the Rayleigh–Sommerfeld integral (within 2%); identification of the focus and of the focal plane; power conservation; merged steps giving the same field; and nonlinear trends with amplitude, single precision, the reference options file, and the CLI.

### `test_kwave.py`

Compares the linear field with k-Wave's continuous-wave angular spectrum solver (`kwave.utils.angular_spectrum_cw`). The same periodic problem must agree to round-off. With k-Wave's zero-padding and angular restriction on a wide domain, the axis and the focal plane agree to 2% of the peak; the difference is pyWave3D's periodic images: 1.3% at 40 mm, 0.9% at 60 mm. pyWave3D is at least as close to the Rayleigh–Sommerfeld integral as k-Wave; both are about 1.3% off, because of the band-limited, hard-edged source. With absorption, the two agree to 1.5%.

It also runs a time-domain k-Wave case (pure-Python backend, a 3.5 mm piston on a 40 × 24 × 24 grid, about 12 s), reduced by `preprocess.time_domain.from_kwave`: its first plane marched by pyWave3D, zero-padded against periodic images, agrees with k-Wave's own axis to 4% of the peak.

It is skipped if `kwave` is not installed. The first import of `kwave` downloads its binaries (about 200 MB).

### `test_iec.py`

Checks the IEC functions against analytic cases: the sinc side lobe at −13.26 dB, Parseval pulse integrals, the $\cos\theta$ obliquity of the exact intensity, zero intensity for evanescent waves, disc averaging, a recovered tilted axis, sawtooth shock parameters, and the MI and TI formulae. It also checks a full run, and that every key named in `docs/iec_outputs.md` is produced.

### `test_preprocess.py`

Records every plane of a nonlinear run and gives the planes back to each preprocessor: as harmonics (also in another axis order, with absolute phase and $e^{-i \omega t}$, and as a 2D single-harmonic field), and as time series in the k-wave-python and fullwave25 formats, sampled at a time step that does not divide the period, after a transient, with drift for FDTD. Each must give back the metrics of the planes (to $10^{-12}$ for harmonics, $10^{-8}$ for time series). It also checks the harmonic fit, the warning for a non-steady record and the input errors.

### `test_beartype.py`

Checks that beartype is on and wraps every annotated function in the package, and that it rejects an argument of the wrong type.

### Other tests

`test_options.py`, `test_model.py` and `test_medium_grid.py` cover the options rules, the input files, the transducer, and the medium and grid. `test_examples.py` checks that the example options file is valid and that the quick example runs work.

## Layout

```
src/pywave3d/
  options.py    JSON options
  medium.py     Medium, Attenuation (water defaults; thermoviscous and power-law absorption)
  grid.py       Grid
  model.py      WaveModel, Transducer, Source (text/binary input, interpolation)
  linear.py     LinearPropagator (diffraction + absorption), AbsorbingLayer (sponge or PML)
  nonlinear.py  NonlinearOperator (spectral RK4 / MUSCL-Godunov), waveform <-> harmonics
  solver.py     WaveSolver (the march), SolverSettings, simulate()
  results.py    Results (recorded data, metrics, files; independent of the solver)
  recorder.py   PlaneRecorder, RecordSettings: Results from planes of harmonics
  preprocess/   spectral.py, time_domain.py (k-wave-python), fdtd.py (fullwave25): other solvers' output to planes
  iec.py        IEC field parameters (derating, MI, TIs, A_ER, side lobes, exact intensity, ...)
  cli.py        the pywave3d command
tests/          pytest suite
docs/           iec_outputs.md
examples/       water_focused.json (the defaults, spelled out), woptions_example.py (runs options/woptions.json)
options/        woptions.json, the reference options file
data/           pxy_at_z0.txt, its source plane
```

## Licence

MIT; see [LICENSE](LICENSE).

# Acoustic field outputs under the IEC standards, and how to compute them by spatial marching

This note lists the acoustic field parameters that the IEC standards for medical and therapeutic ultrasound ask to be determined or declared. For each one, it explains how to compute it in a pseudo-spectral spatial-marching scheme like pyWave3D's, and where pyWave3D records it. pyWave3D records every parameter it can, in every run. The exceptions are pulse duration and the scanned-mode indices, which a single periodic beam cannot represent.

The standards define these quantities for measurements with a hydrophone in water. A simulation gives the same quantities of the modelled field in the modelled medium. The definitions below are paraphrased. Edition, clause and exact wording should be checked against the current text of each standard before anything is declared from a simulation.

## The standards

| Standard | Scope | Main field parameters |
| --- | --- | --- |
| IEC 62127-1 (hydrophone measurement and characterisation of medical ultrasonic fields up to 40 MHz) | the basic definitions used by all the others | p_c, p_r, intensities (instantaneous, pulse-average, temporal-average, spatial-peak), pulse-pressure-squared and pulse-intensity integrals, beam axis, beam widths and areas, acoustic working frequency, pulse duration |
| IEC 61828 (focusing transducers: definitions and measurement methods for the transmitted field) | focused fields | beam maximum, focal plane, focal length (geometric and measured), focal widths, focal area, focal depth (focal length along the beam), focal gain, beam-alignment axis |
| IEC TS 62556 (specification and measurement of high-intensity therapeutic ultrasound (HITU) fields) | HITU transducers, including nonlinear propagation | focal p_c and p_r, focal intensities, −6 dB focal area and depth, power through the focal area, the conditions for extrapolating from low drive |
| IEC 62555 (power measurement for HITU transducers) | total acoustic power | acoustic power (radiation force or calorimetry), and its variation with drive level |
| IEC 61161 (ultrasonic power measurement in liquids up to 40 MHz) | radiation-force balances | total acoustic power |
| IEC 62359 (thermal and mechanical exposure indices) | diagnostic ultrasound | mechanical index MI, thermal indices TIS, TIB, TIC, with the derating of 0.3 dB cm⁻¹ MHz⁻¹ |
| IEC 61689 (physiotherapy systems, 0.5 to 5 MHz) | unfocused physiotherapy transducers | effective radiating area, beam non-uniformity ratio, beam type, effective intensity |
| IEC 60601-2-62 (safety of HITU equipment) | HITU equipment | the output parameters the manufacturer declares, from the standards above |

## How the scheme represents the field

pyWave3D marches the harmonics P_n(x, y), n = 1 … K, of a periodic waveform in retarded time τ = t − z/c₀ from plane to plane. On each plane (z = z₀ + j dz):

- the waveform at any point is the band-limited series p(τ) = Re Σ P_n e^{inωτ}. It is evaluated with an inverse FFT, oversampled (pyWave3D uses 4 samples per period per active harmonic) so that peaks between samples are not missed;
- spatial derivatives and the angular spectrum P̂_n(kx, ky) come from one 2D FFT per harmonic;
- the axial component of the particle velocity of each plane-wave component is v̂_z = P̂_n k_z,n / (ρ₀ n ω), the exact one-way relation.

Everything below uses only these three operations on planes the march already produces, so recording costs no extra propagation. The domain is periodic in x and y and the field is one-way (no reflections), which limits where results are valid: the field must stay away from the lateral edges, or an absorbing layer must be used.

**Other solvers.** The same computations apply to any solver whose output is reduced to such planes of harmonics, and `pywave3d.preprocess` does that reduction:
- frequency-domain pseudo-spectral output is already in this form (`spectral`);
- time-domain pseudo-spectral (k-wave-python) and FDTD (fullwave25) records are fitted for their harmonics over a steady window (`time_domain`, `fdtd`).

`PlaneRecorder` then records every parameter below in the same way. The one-way exact intensity treats each plane's field as travelling in +z, which holds for a beam away from reflectors. Fields from a time-domain solver may also contain reflected waves, so the exact intensity is then an approximation.

**Continuous wave versus pulses.** The scheme as built represents a periodic (CW or tone-burst steady-state) field. For pulsed fields, the "period" must cover the pulse repetition period, or at least a time window containing the whole pulse, with enough harmonics for its bandwidth. Then pulse quantities (pulse integrals, pulse duration, pulse-average intensity) follow from the waveform at each point. For CW fields, pulse-average equals temporal-average.

## Parameters

Status means: **recorded**, pyWave3D computes it in every run and writes it; **not possible**, the scheme cannot represent it (see the last column for why and what is given instead).

Where things are written:

- `metrics.json`, with the IEC parameters under its `iec` key (computed by `pywave3d.iec`, through `Results.iec_metrics()`);
- `axial.txt`, the beam axis;
- `planes.txt`, one line per plane;
- `focal_plane.txt`;
- `results.npz`, which holds every recorded array;
- the point files from `Solver.Output.xyz`.

The options that control these outputs are under `Solver.Metrics`: `focal_quantity`, `levels_dB`, `harmonics`, `hydrophone_diameter`, `derating` and `face_distance`.

### Pressure

| Parameter | Definition (paraphrased) | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Peak-compressional acoustic pressure p_c (p₊) | maximum positive instantaneous pressure at a point (62127-1) | max over τ of the oversampled waveform; per point on the axis, along the x and y lines, over each plane and as the running maximum over z | recorded: `iec.p_c`, `focus.p_positive`, `axial.txt` p+, `planes.txt` max p+, `max_p_plus` |
| Peak-rarefactional acoustic pressure p_r (p₋) | magnitude of the most negative instantaneous pressure (62127-1) | min over τ of the waveform, as above | recorded: `iec.p_r`, `focus.p_negative`, `axial.txt` p−, `planes.txt` min p− |
| Spatial-peak p_c and p_r | maxima of p_c and p_r over the field | running maxima over all planes, with the position of each | recorded: `spatial_peak_p_positive` (x, y, z), `spatial_peak_p_negative` (z) |
| Harmonic amplitudes | amplitude of each frequency component, which shows the degree of nonlinear distortion (62556) | the marched variables themselves, \|P_n\| | recorded: `focus.harmonics`, `axial.txt`, `focal_plane.txt` |
| Acoustic working frequency | arithmetic-mean frequency (the mean of the −3 dB band edges of the spectrum) and zero-crossing frequency (62127-1) | band edges of the line spectrum \|P_n\| at the focus; zero crossings per period of the oversampled waveform at the focus | recorded: `iec.acoustic_working_frequency` |
| Pressure waveform at a point | the time series that the parameters above are computed from | inverse FFT of the harmonics at the nearest grid point, on the nearest plane | recorded: the point files (t, p, p²/ρ₀c₀) |
| Hydrophone-averaged values | what a hydrophone with a sensitive element of finite diameter reports (62127-1 spatial averaging) | the harmonics of the focal plane averaged over a disc of that diameter (FFT convolution), then p_c, p_r and \|P_n\| at the focus | recorded when `Metrics.hydrophone_diameter` > 0: `iec.hydrophone` |

### Intensity and power

| Parameter | Definition (paraphrased) | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Instantaneous intensity | p v; in measurements taken as p²/(ρ₀c₀) (62127-1) | p(τ)²/(ρ₀c₀) from the waveform | recorded: third column of the point files |
| Temporal-average intensity I_ta | average of the instantaneous intensity over the repetition period | plane-wave form Σ_n \|P_n\|²/(2ρ₀c₀), as hydrophone measurements report it. Exact active intensity along z, Σ_n ½ Re(P_n v_z,n*), with v_z,n from the angular spectrum (`iec.active_intensity`); the two differ by the obliquity of the waves | recorded: `iec.I_ta_focus` and `iec.I_ta_focus_exact`, `axial.txt` I, `focal_plane.txt` I |
| Spatial-peak temporal-average intensity I_spta | maximum of I_ta over the field | maximum of I_ta over each plane, then over z | recorded: `iec.I_spta`, `spatial_peak_intensity` (value, z), `planes.txt` max I, `iec.I_spta_focal_plane_exact` |
| Pulse-pressure-squared integral (ppsi) and pulse-intensity integral (pii) | ∫ p² dt and ∫ p²/(ρ₀c₀) dt over the pulse | over one period by Parseval: T Σ \|P_n\|²/2 and that over ρ₀c₀ | recorded per period: `iec.ppsi_focus`, `iec.pii_focus`, `iec.pii_spatial_peak`, `axial.txt` pii |
| Pulse-average intensity I_pa, and I_sppa | pii divided by the pulse duration, and its spatial peak | for a continuous wave, I_pa = I_ta | recorded: `iec.I_sppa` (= I_spta, as the field is continuous) |
| Pulse duration, pulse repetition period | 1.25 × the time between 10% and 90% of the ppsi | **not possible**: the scheme marches one period of a periodic (CW) field, so the field has no pulse. A tone burst long enough to reach steady state is the CW field. A short pulse would need the period to cover the whole repetition period, with enough harmonics for its bandwidth | `iec.field` states that the field is CW |
| Acoustic output power P | total power through a plane (61161, 62555) | plane-wave estimate Σ_n Σ_{x,y} \|P_n\|² dA/(2ρ₀c₀) on every plane; exact power Σ ½ Re(P̂ v̂_z*) dA (evanescent components carry none) on the source, focal and last planes | recorded: `planes.txt` power, `iec.power` (`source`, `focal_plane`, `end`, and `exact`) |
| Power through the focal area | the part of the power that reaches the focus (62556) | sum of I_ta dA over the connected region of the focal plane above the first of `levels_dB` | recorded: `iec.power_in_focal_area_6dB` and `..._exact` |

### Beam geometry and focus (IEC 61828, 62556)

| Parameter | Definition (paraphrased) | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Beam axis | the line through the beam centre on planes along z (beam-alignment axis) | the intensity centroid of the −6 dB region of every plane, then a least-squares line through them. The profiles are recorded on the transducer axis (the plane centre) | recorded: `planes.txt` beam centre x, y; `iec.beam_alignment_axis` (intercepts, slopes, tilt); `axial.txt` along the transducer axis |
| Beam maximum / focus | point of the maximum of the chosen quantity on the beam axis (pressure focus or intensity focus) | the focal quantity on the axis is compared at every recorded plane with the best so far, and that plane is kept (`focal_quantity`: `p_positive`, `p_negative` or `first_harmonic`). The intensity maximum is reported too | recorded: `focus` (x, y, z, index, p₊, p₋, harmonics, intensity, heating rate), `axis_peak_intensity` |
| Focal plane | the plane through the focus, perpendicular to the axis | the copy of the harmonics on the plane of the focus | recorded: `focal_plane.txt`, `focal_plane` in `results.npz` |
| Plane of the spatial peak | where the field maximum lies if it is not on the axis | as the focal plane, for the plane maximum of p₊ | recorded: `peak_plane`, `spatial_peak_p_positive.off_axis` |
| Focal length, beam-maximum distance, focal shift | distance from the source to the focus, and its difference from the geometric focal length | z of the focus minus z₀, minus the radius of curvature | recorded: `focus.z`, `focal_shift` |
| Focal widths (−3, −6, −12, −20 dB) | width of the beam in the focal plane at the level, in two orthogonal directions | linear interpolation of the level crossings along the x and y lines through the focus | recorded at every level in `levels_dB`: `focal_width_x_6dB`, `focal_width_y_6dB`, … |
| Focal depth | extent along the axis above the level | level crossings of the axial profile around the focus; NaN if the region reaches the source plane or the end of the domain | recorded: `focal_length_6dB`, … |
| Focal area, beam area | area in the focal plane above the level | grid cells of the connected region around the focus, times dx dy | recorded: `focal_area_6dB`, … |
| Focal gain | focal pressure over source pressure | \|P_1\| at the focus over the source amplitude | recorded: `focal_gain` |
| Side lobes, prefocal and post-focal maxima | secondary maxima, for example for safety near skin and bone | the largest maxima outside the main lobe. The lobe extends past the −6 dB level to its first minimum, so ripples on it do not count. Computed along the axis (prefocal, post-focal) and along x and y in the focal plane | recorded: `iec.side_lobes` (`axial`, `x`, `y`: value, position, level in dB) |

### Exposure indices (IEC 62359)

| Parameter | Definition (paraphrased) | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Derated quantities (p_r,α, I_ta,α, P_α) | water values reduced by 0.3 dB cm⁻¹ MHz⁻¹ along the axis | multiply by 10^(−a f_awf z / 20) for pressures and by its square for intensity and power (`iec.derating`; a = `Metrics.derating`). This is the standard's linear derating of the water field. For derated *nonlinear* propagation, run in a medium with that attenuation instead | recorded: `axial.txt` p_r,α and I_ta,α; `iec.derated.I_spta_derated_axis`, `p_r_derated_max_axis` |
| Mechanical index MI | p_r,α (in MPa) at the depth z_MI of the maximum of the derated pii (beyond the break-point depth z_bp), divided by √(f_awf in MHz) | from the axial p_r and pii, derated, with f_awf from the focus | recorded: `iec.derated.MI`, `z_MI`, `p_r_derated`; the value without derating is `focus.mechanical_index_water` |
| Thermal indices TIS, TIB, TIC (unscanned mode) | TIS = P f / C_TIS for an aperture ≤ 1 cm², else max over z ≥ z_bp of min(P_α, I_ta,α × 1 cm²) f / C_TIS; TIB = max over z ≥ z_bp of min(√(P_α I_ta,α)/C_TIB1, P_α/C_TIB2); TIC = P/(D_eq C_TIC), with D_eq = √(4 A_aprt/π) and z_bp = 1.5 D_eq | P from the source plane; I_ta,α along the axis; A_aprt is the −12 dB output beam area of the source plane; C_TIS = 210 mW MHz, C_TIB1 = 50 mW/cm, C_TIB2 = 4.4 mW, C_TIC = 40 mW/cm (`iec.thermal_indices`). The formulae and constants are for diagnostic equipment: check them against the edition in force | recorded: `iec.derated.TIS`, `TIB`, `TIC`, `A_aprt`, `D_eq`, `z_bp` |
| Scanned-mode indices | the same for scanning beams | **not possible** with one beam: they need the field of every scan line | — |
| Heating rate q | (not an IEC index; used to plan HITU treatments) 2 Σ α_n I_n | from the harmonics and the absorption α_n of the medium | recorded: `axial.txt` q, `focus.heating_rate` |

### Physiotherapy (IEC 61689)

| Parameter | Definition (paraphrased) | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Beam cross-sectional area A_BCS | smallest area holding 75% of the mean-square pressure, on a plane near the face | on the plane kept at `face_distance` (default 0.3 cm): sort the intensity and add up the smallest area holding 75% (`iec.area_holding`) | recorded: `iec.physiotherapy.A_BCS`, `z` |
| Effective radiating area A_ER | A_BCS at 0.3 cm times a factor (1.333) | F_AC × A_BCS | recorded: `iec.physiotherapy.A_ER` |
| Effective intensity | output power over A_ER | P / A_ER | recorded: `iec.physiotherapy.effective_intensity` |
| Beam non-uniformity ratio R_BN | spatial-peak intensity relative to the mean intensity over A_ER | I_spta / (P / A_ER), in the plane-wave intensity form | recorded: `iec.physiotherapy.R_BN` |

### Nonlinear and shock parameters (IEC TS 62556 conditions)

| Parameter | Use | Computation in the marching scheme | Status and where |
| --- | --- | --- | --- |
| Degree of nonlinear distortion | whether the field is nonlinear, and so whether low-drive extrapolation is valid (62556) | p_c/p_r and \|P_2\|/\|P_1\| at the focus; a second run at another amplitude (or linear) shows the departure from linear scaling | recorded: `iec.nonlinear.p_c_over_p_r`, `P2_over_P1` |
| Steepness, shock formation, shock amplitude | where shocks form and how large they are (histotripsy, boiling histotripsy) | max dp/dt over ω\|P_1\| (1 for a sinusoid); the rise across the steepest part over T/K; the z where the Godunov scheme takes over | recorded: `iec.nonlinear.steepness`, `shock_amplitude`, `shocked`, `godunov_from_z` |

## Practical notes

- **Resolution.** The −6 dB widths need at least about 4 points across the focal width, and the axial maxima about 4 planes over the focal depth. The peak pressures of shocked waveforms depend on the number of harmonics: p₊ converges slowly, so check it with `SamplingFrequencyMultiplier` doubled.
- **Spatial averaging.** A hydrophone averages over its sensitive element. Set `Metrics.hydrophone_diameter` to compare focal values with a measurement.
- **Low-drive extrapolation.** 62556 lets HITU focal values be extrapolated from low-drive measurements when the field is close to linear. A linear run (`"NLN": false`) gives that extrapolation, and the nonlinear run gives the actual values. Comparing the two quantifies the error of extrapolating.

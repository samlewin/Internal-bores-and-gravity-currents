This folder contains all of the code and data required to produce Figures 2 and 3 of the manuscript. The plotting code is written in Python (v3.10.12) for Jupyter Notebooks, using numpy (v3.7.2) and matplotlib (v1.25.2). Data is saved in Python .npz and MATLAB .mat format. Saved fields are detailed below.

# Bore and wave-speed statistics

Arrays use **mooring × bore-slot** ordering, normally **42 × 164**. 

Background profiles use the **30 minutes before arrival**. Bore diagnostics use the **first 60 minutes after arrival**. Temporal smoothing, where applied, is a **31-point, second-order Savitzky–Golay filter**.

Missing, unselected or unsuccessful numeric results are stored as `NaN`.

## `bore_stats`

| Field | Description | Units |
|---|---|---|
| `H` | Total water depth | m |
| `A` | Pre-arrival buoyancy contrast, \(g'=b_U-b_L\); amplitude of the fitted buoyancy profile | m s⁻² |
| `C` | Vertical location parameter of the fitted tanh buoyancy profile, measured above bed | m |
| `lam` | Thickness parameter of the fitted tanh profile | m |
| `S` | Ambient buoyancy contrast relative to the surface-intrusion buoyancy contrast; values below 1 indicate excess surface buoyancy | Dimensionless |
| `U` | Estimated bore speed: maximum temporally smoothed, rotated velocity averaged over a band around the maximum mean eastward velocity anomaly | m s⁻¹ |
| `etaB` | Maximum temporally smoothed depression of the sampled interior background isopycnals | m |
| `etaGC` | Maximum temporally smoothed depression of the pre-arrival surface isopycnal | m |
| `APE` | Mean column-integrated available potential energy, normalized by \(\rho_0g'H^2\) | Dimensionless |
| `KE` | Mean column-integrated kinetic energy of the along-wave velocity component, with the same normalization | Dimensionless |
| `max_depression` | Largest instantaneous downward displacement; used for saturation classification | m |
| `KE_v` | Mean column-integrated kinetic energy of the transverse velocity component, with the same normalization | Dimensionless |
| `buoyancy_max_displacement` | Buoyancy at the location and time of maximum absolute instantaneous displacement, relative to the pre-arrival bottom reference | m s⁻² |
| `theta` | Rotation angle calculated from the background-subtracted velocity components near the maximum mean eastward anomaly | radians |
| `DeltaGC` | Gravity-current index: \((\eta_{GC}/\eta_B)^2(1-S)/S\) | Dimensionless |
| `datetime` | Arrival date and time | Datetime |
| `mooring` | Mooring identifier | Text |

Velocity diagnostics use background subtraction along isopycnals. Displacement amplitudes are positive for depression. `max_depression` is instantaneous, whereas `etaB` and `etaGC` use temporal smoothing.

## `djl_stats`

| Field | Description | Units |
|---|---|---|
| `c0` | Mode-1 linear long-wave speed for the pre-arrival stratification | m s⁻¹ |
| `alpha` | KdV quadratic nonlinearity coefficient from the linear structure function | s⁻¹ |
| `cDJL` | Accepted conjugate-flow propagation speed | m s⁻¹ |
| `etaDJL` | Maximum absolute displacement of the accepted conjugate-flow profile | m |
| `max_gradient` | Displacement gradient at the upper integration boundary: the surface for no-core solutions, or core boundary for smooth-core solutions | Dimensionless |
| `grid_shape` | Full output dimensions, normally `[42, 164]` | Two integers |
| `solver` | Successful branch: `no_core` or `smooth_core`; empty if none succeeded | Text |
| `processed` | Whether the batch runner finished handling that arrival, including skipped inputs and failed solves | Boolean |

`c0` and `alpha` can remain valid when both nonlinear solvers fail. 

Negative `alpha` favours small-amplitude depression waves and positive `alpha` favours elevation waves. This is not an existence test for finite-amplitude solutions.

All saved speeds and displacements are **dimensional**.

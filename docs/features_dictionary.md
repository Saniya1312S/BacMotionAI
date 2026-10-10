# Features Dictionary

This dictionary defines the 14 features extracted from the bacterial trajectories in the BacMotionAI simulation.

| Feature Name | Description | Formula / Computation | Unit | Typical Range | Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`avg_speed`** | The mean scalar speed of the bacterium over the entire trajectory. | `mean(Δr / Δt)` | µm/s | 5 - 25 | Speed |
| **`max_speed`** | The maximum instantaneous speed observed during the trajectory. | `max(Δr / Δt)` | µm/s | 15 - 35 | Speed |
| **`tumble_freq`** | The frequency of tumbling events per second. | `Count(Δθ > threshold) / Total Time` | Hz | 0.1 - 2.0 | Motility |
| **`mean_run_length`** | The average distance covered during a continuous "run" phase (between tumbles). | `mean(Distance between tumbles)` | µm | 10 - 50 | Motility |
| **`msd`** | Mean Squared Displacement at a specific lag time. | `<\|r(t + τ) - r(t)\|²>` | µm² | 100 - 2000 | Transport |
| **`tortuosity`** | The ratio of the actual path length to the straight-line distance between start and end points. | `Total Path Length / Linear Distance` | Dimensionless | 1.1 - 5.0 | Geometry |
| **`wall_time_ratio`** | The fraction of total time spent within the hydrodynamic interaction zone near the channel walls. | `Time(y < h_thresh or y > W - h_thresh) / Total Time`| Dimensionless | 0.0 - 0.9 | Wall |
| **`orientation_anisotropy`** | A measure of how aligned the bacterial motion is with the channel axis. | `<cos²(θ)>` | Dimensionless | 0.0 - 1.0 | Orientation |
| **`chemotactic_index`** | The component of velocity directed along the chemical gradient relative to the average speed. | `V_drift / avg_speed` | Dimensionless | -0.5 - 0.5 | Transport |
| **`peclet_number`** | The ratio of advective transport rate (swimming) to diffusive transport rate (tumbling). | `(avg_speed * Length) / D_eff` | Dimensionless | 10 - 1000 | Transport |
| **`confinement_ratio`** | The ratio of the bacterial run length to the channel width. | `mean_run_length / Channel_Width` | Dimensionless | 0.1 - 2.0 | Geometry |
| **`turn_angle_variance`** | The variance of the change in heading angle during tumbling events. | `var(Δθ_tumble)` | rad² | 0.5 - 2.5 | Orientation |
| **`hydro_trap_duration`** | The maximum continuous time spent trapped at the wall boundary. | `max(Continuous Wall Time)` | s | 0 - 10 | Wall |
| **`drift_velocity`** | The mean velocity component parallel to the applied stimulus gradient. | `mean(v_x)` (if gradient is along x) | µm/s | -10 - 10 | Speed |

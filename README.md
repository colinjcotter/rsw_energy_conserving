# RSW energy-conserving simulation

Rotating shallow water on the sphere with an energy-conserving cPG time
discretisation and optional stochastic (SFLT) forcing, built on Firedrake and
IRKsome.

- `poisson_energy.py` — the script you run
- `poisson_tools.py` — noise, equations, solvers
- `sw_tools.py` — mesh, spaces, initial conditions, command-line arguments

## Running

With Firedrake and IRKsome available:

    mpiexec -n 16 python poisson_energy.py --williamson 6 --ref_level 4 \
        --nsteps 400 --tmax 172800 --seed 1 --no_output

Stochastic forcing is on by default; `--pure` turns it off. `--williamson 2` is
Läuter et al. (2005) Example 3, which has an exact solution.
`python poisson_energy.py --help` lists all options.

Each run saves the final-time velocity and depth to
`../RSW_checkpoint/new_RSW_checkpoint/`, and the per-step energy error and
divergence norm to `w<case>_energy_errors_<SFLT|pure>.txt` and
`w<case>_div_norms_<SFLT|pure>.txt` in the current directory (these are
overwritten by the next run of the same case).

## Convergence studies with shared noise

Runs being compared must see the same noise (same `--seed`).

- **Spatial** (vary `--ref_level`, same `dt`): run the finest mesh first with
  `--ref_level_fine` equal to its `--ref_level`; it saves the noise. Coarser
  meshes use the same `--ref_level_fine`, load the noise and interpolate it.
- **Temporal** (same mesh, vary `--nsteps`): the finest run saves every `dW`;
  a run with k times fewer steps loads them and sums k fine `dW` per step.
  Use one `--noise_dir` per seed.

      mpiexec -n 16 python poisson_energy.py --nsteps 6400 --save_noise --noise_dir ./noise_checkpoints/seed1 ...
      mpiexec -n 16 python poisson_energy.py --nsteps 1600 --load_noise --coarsening 4 --noise_dir ./noise_checkpoints/seed1 ...

Then, from the same directory, with arguments matching the runs:

- `scnv_analysis.py` — spatial convergence against the finest mesh
- `tcnv_analysis.py` — temporal convergence against the finest `dt`
- `exact_cnv_analysis.py`, `exact_tcnv_analysis.py` — spatial and temporal
  error against the exact solution (`--williamson 2 --pure` runs)

Add `--SFLT` to the first two for stochastic runs.

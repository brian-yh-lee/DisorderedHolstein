# DisorderedHolstein.jl

Exact diagonalization + Metropolis-adjusted Langevin algorithm Monte Carlo sampling for a disordered Holstein model in the zero phonon frequency limit

$$H = -\sum_{ij\sigma} t_{ij} c_{i\sigma}^\dagger c_{j\sigma} + \frac{1}{2} K \sum_i x_i^2 + \sum_i \gamma_i x_i (n_i - \nu)$$

where $t_{ij}$ is the hopping matrix, $c_{i\sigma}$ is a electronic annihilation operator, $x_i$ is the phonon coordinate operator, $K$ is the phonon spring constant, $\gamma_i$ is the electron-phonon coupling, $n_i = \sum_\sigma c_{i\sigma}^\dagger c_{i\sigma}$, and $\nu$ is the spinful electronic filling. We take $\gamma_i$ to be a classical random variable.  

The module allows for numerically exact computation of optical conductivity, spectral functions, and other single quasiparticle quantities. This module was developed and used to support the claims made in ["Apparent Planckian scattering from local polaron formation"](https://arxiv.org/abs/2604.22029). An example of running a simulation is provided in `example.jl`. 

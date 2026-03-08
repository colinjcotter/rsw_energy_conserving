import numpy as np
import matplotlib.pyplot as plt

# Load the energy errors data from the text file
energy_errs_sflt = np.loadtxt("w6_energy_errors_SFLT_3.txt")
energy_errs = np.loadtxt("w6_energy_errors_pure_3.txt")

# Create the plot
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(10, 6))

# Handle zero values by adding small epsilon or masking
energy_errs_plot = np.where(energy_errs == 0, 1e-16, np.abs(energy_errs))
energy_errs_sflt_plot = np.where(energy_errs_sflt == 0, 1e-16, np.abs(energy_errs_sflt))

plt.semilogy(energy_errs_plot, 'b-', linewidth=2, label='Deterministic')
plt.semilogy(energy_errs_sflt_plot, 'r-', linewidth=2, label='Stochastic')

#plt.axhline(y=1e-15, color='k', linestyle='--', alpha=0.5, label='Machine precision')

plt.xlabel('Time steps')
plt.ylabel('Relative Energy error (log scale)')
plt.title('Energy Conservation: Relative Energy Error vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(1e-16, 1e-10)  # Extend range lower

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Data shape: {energy_errs.shape}")
print(f"Total points: {len(energy_errs)}")
print(f"Min error: {np.min(energy_errs):.6e}")
print(f"Max error: {np.max(energy_errs):.6e}")
print(f"Mean error: {np.mean(energy_errs):.6e}")
print(f"Final error: {energy_errs[-1]:.6e}")

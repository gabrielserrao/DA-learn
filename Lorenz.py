import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import grad

# Define the Lorenz system
def lorenz(t, y, params):
    sigma, rho, beta = params
    return torch.tensor([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]])



############################################################################################################
#4DVar
############################################################################################################
# Define the cost function for the 4DVar optimization
def cost_func(model_output, obs, params, pb, B_inv):
    obs_term = torch.sum((model_output - obs) ** 2)  # Observation term: sum of squared differences
    bg_term = torch.matmul((params - pb).T, B_inv @ (params - pb))  # Background term: weighted departure of parameters from prior estimates
    cost = 0.5 * obs_term + 0.5 * bg_term  # Total cost is a combination of the observation term and the background term
    return cost

# Define the true parameters and generate synthetic observations from the Lorenz system
true_params = torch.tensor([10., 28., 8./3])
t_values = torch.linspace(0, 2, 100)
y0 = torch.tensor([-8., 8., 27.])
obs = odeint(lambda t, y: lorenz(t, y, true_params), y0, t_values) + 0.1 * torch.randn(t_values.shape[0], 3)


# Define our initial guess for the parameters and our prior guess
params = torch.tensor([8., 20., 2.], requires_grad=True)  # Initial guess
pb = torch.tensor([9., 26., 2.5])  # Prior guess

# Define our background error covariance matrix (assumed to be the identity in this case)
B = torch.eye(3)
B_inv = torch.inverse(B)

# Define our optimizer (Adam in this case)
num_iterations=1000
optimizer = torch.optim.Adam([params], lr=0.01)

# Initialization of the list for storing outputs at each iteration
outputs = []
costs = []
# Optimization loop
for i in range(num_iterations):
    optimizer.zero_grad()  # Clear gradients from previous step
    model_output = odeint(lambda t, y: lorenz(t, y, params), y0, t_values)

    cost = cost_func(model_output, obs, params, pb, B_inv)  # Calculate the cost function
    cost.backward()  # Calculate the gradients
    optimizer.step()  # Update parameters using the gradients

    # Store the model output at this iteration
    outputs.append(model_output.detach())
    costs.append(cost.item())
    
    # Print the estimated and true parameters
print(f'Estimated parameters 4DVar: {params.detach().numpy()}')
print(f'True parameters: {true_params.numpy()}')

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

variables = ['x', 'y', 'z']

for i, variable in enumerate(variables):
    for output in outputs[:-1]:
        axs[i].plot(t_values.numpy(), output[:, i].numpy(), color='lightgray')

    axs[i].plot(t_values.numpy(), outputs[-1][:, i].numpy(), color='blue', label='Estimated')
    axs[i].plot(t_values.numpy(), obs[:, i], label='True',color='red', marker = 'o', linestyle = 'None')

    axs[i].set_xlabel('Time')
    axs[i].set_ylabel(variable)
    axs[i].legend()

plt.tight_layout()
plt.show()



############################################################################################################
# Metropolis-Hastings MCMC
############################################################################################################
# Prior Distribution
def prior_distribution():
    prior_sigma = torch.distributions.Uniform(0.0, 20.0)
    prior_rho = torch.distributions.Uniform(0.0, 40.0)
    prior_beta = torch.distributions.Uniform(0.0, 10.0)
    return torch.tensor([prior_sigma.sample(), prior_rho.sample(), prior_beta.sample()])

# Likelihood Function
def likelihood_function(model_output, obs):
    likelihood = torch.distributions.Normal(model_output, 0.1).log_prob(obs).sum()
    return likelihood

# Metropolis-Hastings
def metropolis_hastings(obs, num_iterations):
    parameter_chain = []
    current_parameter = prior_distribution()

    for i in range(num_iterations):
        proposed_parameter = torch.distributions.Normal(current_parameter, 0.1).sample()
        model_output_current = odeint(lambda t, y: lorenz(t, y, current_parameter), y0, t_values)
        model_output_proposed = odeint(lambda t, y: lorenz(t, y, proposed_parameter), y0, t_values)


        current_likelihood = likelihood_function(model_output_current, obs)
        proposed_likelihood = likelihood_function(model_output_proposed, obs)

        acceptance_ratio = torch.exp(proposed_likelihood - current_likelihood)

        if acceptance_ratio > torch.rand(1):
            current_parameter = proposed_parameter

        parameter_chain.append(current_parameter)

    return parameter_chain

# Run MCMC
parameter_chain = metropolis_hastings(obs, num_iterations=1000)

# Postprocessing
parameter_chain = torch.stack(parameter_chain)
posterior_mean = parameter_chain.mean(dim=0)

print(f'Estimated parameters (MCMC): {posterior_mean.detach().numpy()}')
plt.plot(parameter_chain[:, 0].numpy(), label='Sigma')
plt.plot(parameter_chain[:, 1].numpy(), label='Rho')
plt.plot(parameter_chain[:, 2].numpy(), label='Beta')
plt.axhline(true_params[0].item(), color='red', linestyle='--')
plt.axhline(true_params[1].item(), color='blue', linestyle='--')
plt.axhline(true_params[2].item(), color='green', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Parameter Values")
plt.legend()
plt.show()

############################################################################################################
#HMC
############################################################################################################

# HMC parameters

initial_velocity = torch.randn(3)
num_samples = 10
step_size = 0.01
num_steps = 100

# Define the leapfrog integrator for HMC
def leapfrog(position, velocity, potential_energy, step_size, num_steps):
    velocity -= 0.5 * step_size * grad(potential_energy(position), position)[0]
    for _ in range(num_steps - 1):
        position = position + step_size * velocity
        velocity -= step_size * grad(potential_energy(position), position)[0]
    velocity -= 0.5 * step_size * grad(potential_energy(position), position)[0]
    return position, velocity

# Define the kinetic energy for HMC
def kinetic_energy(velocity):
    return 0.5 * velocity.dot(velocity)

# Define the Hamiltonian for HMC
def hamiltonian(position, velocity):
    potential_energy = cost_func(odeint(lambda t, y: lorenz(t, y, position.detach()), y0, t_values), obs, position, pb, B_inv)
    return potential_energy + kinetic_energy(velocity)

def potential_energy_wrapper(position):
    return hamiltonian(position, initial_velocity) - kinetic_energy(initial_velocity)


# Hamiltonian Monte Carlo (HMC)
samples = []
for _ in range(num_samples):
   # Perform leapfrog integration
    new_params, new_velocity = leapfrog(params, initial_velocity, potential_energy_wrapper, step_size, num_steps)
    
    
    # Compute Hamiltonians at the start and end of the trajectory
    start_hamiltonian = hamiltonian(params, initial_velocity)
    new_hamiltonian = hamiltonian(new_params, new_velocity)
    
    # Metropolis-Hastings acceptance
    if torch.rand(1) < torch.exp(start_hamiltonian - new_hamiltonian):
        params = new_params
        initial_velocity = torch.randn(3)  # Resample velocity
    
    samples.append(params.detach())

# Convert to a tensor for convenience
samples = torch.stack(samples)

# Compute the mean of the samples
mean_params = torch.mean(samples, axis=0)

# Print the estimated and true parameters
print(f'Estimated parameters (HMC): {mean_params.numpy()}')
print(f'True parameters: {true_params.numpy()}')

# Plot the parameter samples
plt.plot(samples[:, 0].numpy(), label='Sigma')
plt.plot(samples[:, 1].numpy(), label='Rho')
plt.plot(samples[:, 2].numpy(), label='Beta')
plt.axhline(true_params[0].item(), color='red', linestyle='--')
plt.axhline(true_params[1].item(), color='blue', linestyle='--')
plt.axhline(true_params[2].item(), color='green', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Parameter Values")
plt.legend()
plt.show()


############################################################################################################
#PLOTING
############################################################################################################
# Additional Plots


fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for i, variable in enumerate(variables): 
    axs[i].plot(t_values.numpy(), odeint(lambda t, y: lorenz(t, y, posterior_mean), y0, t_values).detach()[:, i].numpy(), color='blue', label='Estimated MCMC')
    axs[i].plot(t_values.numpy(), outputs[-1][:, i].numpy(), color='green', label='Estimated 4DVar')
    axs[i].plot(t_values.numpy(), obs[:, i], label='True',color='red', marker = 'o', linestyle = 'None')
    axs[i].plot(t_values.numpy(), odeint(lambda t, y: lorenz(t, y, mean_params), y0, t_values).detach()[:, i].numpy(), color='orange', label='Estimated HMC')

    axs[i].set_xlabel('Time')
    axs[i].set_ylabel(variable)
    axs[i].legend()

plt.tight_layout()
plt.show()

# Parameter estimates
params_4dvar = params.detach().numpy()
params_mcmc = posterior_mean.detach().numpy()
params_hmc = mean_params.numpy()

# True parameters
true_params_np = true_params.numpy()

# Parameter names
param_names = ['Sigma', 'Rho', 'Beta']

# Create scatter plots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, ax in enumerate(axs):
    # 4DVar estimates
    ax.scatter([1], params_4dvar[i], marker='o', color='b', s=100, label='4DVar')
    
    # MCMC estimates
    ax.scatter([2], params_mcmc[i], marker='o', color='r', s=100, label='MCMC')
    
    # HMC estimates
    ax.scatter([3], params_hmc[i], marker='o', color='g', s=100, label='HMC')
    
    # True parameters
    ax.plot([1, 2, 3], [true_params_np[i]] * 3, color='k', linestyle='--', label='True')
    ax.text(1.5, true_params_np[i], f'{param_names[i]}: {true_params_np[i]}', va='bottom', ha='center')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['4DVar', 'MCMC', 'HMC'])
    ax.set_ylabel(param_names[i])
    ax.legend()
    ax.grid(False)

plt.tight_layout()
plt.grid(False)
plt.show()


# Plot trajectories
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 4DVar trajectory
trajectory_4dvar = odeint(lambda t, y: lorenz(t, y, torch.tensor(params_4dvar)), y0, t_values).detach().numpy()
ax.plot(trajectory_4dvar[:, 0], trajectory_4dvar[:, 1], trajectory_4dvar[:, 2], label='4DVar')

# MCMC trajectory
trajectory_mcmc = odeint(lambda t, y: lorenz(t, y, torch.tensor(params_mcmc)), y0, t_values).detach().numpy()
ax.plot(trajectory_mcmc[:, 0], trajectory_mcmc[:, 1], trajectory_mcmc[:, 2], label='MCMC')

# HMC trajectory
trajectory_hmc = odeint(lambda t, y: lorenz(t, y, torch.tensor(params_hmc)), y0, t_values).detach().numpy()
ax.plot(trajectory_hmc[:, 0], trajectory_hmc[:, 1], trajectory_hmc[:, 2], label='HMC')

# True trajectory
true_trajectory = odeint(lambda t, y: lorenz(t, y, true_params), y0, t_values).detach().numpy()
ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], label='True')

ax.legend()
plt.show()


import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.patches import Circle
import numpy as np

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example setup: known agent 1 trajectory (T timesteps, 2D positions)
T = 10
dt = torch.tensor(0.1, device=device)



# === Define Time-Varying System Matrices ===
def A_t(t):
    return torch.tensor([[-0.1 * t, 0], [0, -0.1 * t]], device=device)

def B_t(t):
    return torch.tensor([[0.5 + 0.05 * t, 0], [0, 0.5 + 0.05 * t]], device=device)


# Define agent 1 trajectory
x_vals = torch.linspace(2, 0, T, device=device)
y_vals = torch.linspace(0, 1, T, device=device)

agent1_traj = torch.stack((x_vals, y_vals), dim=1)

# Define start and goal constraints for agent2
waypoints = {
    0: (torch.tensor([0.0, 0.0], device=device), 0.1),
    T-1: (torch.tensor([2.0, 1.0], device=device), 0.1)
}

# Define the generative model with masked collision-avoidance and start/goal constraints
def model(agent1_obs):
    T, D = agent1_obs.shape
    X = pyro.sample("X", dist.Normal(torch.zeros(T, D, device=device), torch.ones(T, D, device=device)).to_event(2))
    safe_distance = torch.tensor(1, device=device)  # Minimum collision-free distance
    mask_threshold = torch.tensor(1, device=device)  # Only apply collision penalty if within this threshold

    for t in range(T - 1):
        A = A_t(t)
        B = B_t(t)

        # Ensure epsilon_t is sampled on GPU
        epsilon_t = pyro.sample(f"epsilon_{t}", dist.Normal(torch.tensor(0.0, device=device),
                                                             torch.tensor(1.0, device=device)).expand([2]).to_event(1))

        # Deterministic part of the SDE transition
        x_next_pred = X[t] + A @ X[t] * dt

        # Stochastic part (diffusion term) - now fully on GPU
        diffusion = B @ epsilon_t * torch.sqrt(dt)

        # Penalize deviations from the expected SDE transition
        deviation = (X[t+1] - x_next_pred - diffusion) ** 2
        pyro.factor(f"sde_constraint_{t}", -0.5 * torch.sum(deviation) / (0.1 ** 2))

        # Penalize getting too close to obstacles

    # Calculate distance and apply collision penalty with masking
    distance = torch.norm(X - agent1_obs, dim=-1)
    mask = distance < mask_threshold
    collision_penalty = torch.where((distance < safe_distance) & mask,
                                    (safe_distance - distance) ** 2,
                                    torch.zeros_like(distance, device=device))
    pyro.factor("collision_penalty", -collision_penalty.sum()*1e2)

    # Factor for start and goal constraint penalties
    for t, (mean, var) in waypoints.items():
        pyro.factor(f"waypoint_{t}", -1e6 * torch.sum(((X[t] - mean) ** 2) / var))

# Define the guide (variational distribution)
def guide(agent1_obs):
    T, D = agent1_obs.shape
    loc = pyro.param("loc", torch.zeros(T, D, device=device))
    scale = pyro.param("scale", torch.ones(T, D, device=device), constraint=pyro.distributions.constraints.positive)
    pyro.sample("X", dist.Normal(loc, scale).to_event(2))

# Setup the optimizer and SVI
pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=JitTrace_ELBO())

# Run inference
num_steps = 5000
for step in range(num_steps):
    loss = svi.step(agent1_traj)
    if step % 100 == 0:
        print(f"Step {step} : loss = {loss}")

# Retrieve inferred mean trajectory for agent2
inferred_loc = pyro.get_param_store()["loc"].detach().cpu().numpy()

print("Inferred agent2 trajectory (collision-avoiding with masking, start and goal constrained):")
print(inferred_loc)

# Visualization
x2, y2 = inferred_loc[:, 0], inferred_loc[:, 1]
points2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

norm2 = plt.Normalize(0, len(segments2))
colors2 = plt.cm.viridis(norm2(range(len(segments2))))

lc2 = mcoll.LineCollection(segments2, cmap='viridis', norm=norm2, linewidth=2)
lc2.set_array(np.arange(len(segments2)))

x1, y1 = agent1_traj[:, 0].cpu().numpy(), agent1_traj[:, 1].cpu().numpy()
points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)

norm1 = plt.Normalize(0, len(segments1))
colors1 = plt.cm.plasma(norm1(range(len(segments1))))

lc1 = mcoll.LineCollection(segments1, cmap='plasma', norm=norm1, linewidth=2)
lc1.set_array(np.arange(len(segments1)))

fig, ax = plt.subplots()
ax.add_collection(lc1)
ax.add_collection(lc2)

ax.plot(waypoints[0][0].cpu()[0], waypoints[0][0].cpu()[1], marker="o", color="green", label="Agent2 Start")
ax.plot(waypoints[T-1][0].cpu()[0], waypoints[T-1][0].cpu()[1], marker="o", color="red", label="Agent2 Goal")

ax.autoscale()
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(lc2, label="Agent2 Time Progression")
plt.colorbar(lc1, label="Agent1 Time Progression")
plt.legend()
plt.savefig("results/multi_agent.png")
plt.close()
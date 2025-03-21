import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam

# === GPU Setup ===
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Time Grid ===
T = 10  # Number of time steps
dt = torch.tensor(0.1, device=device)  # Time step size (explicitly on GPU)

# === Define Time-Varying System Matrices ===
def A_t(t):
    return torch.tensor([[-0.1 * t, 0], [0, -0.1 * t]], device=device)

def B_t(t):
    return torch.tensor([[0.5 + 0.05 * t, 0], [0, 0.5 + 0.05 * t]], device=device)

# === Define Waypoints (Time Index, (x, y), Variance) ===
waypoints = {
    0: (torch.tensor([0.0, 0.0], device=device), 0.1),
    9: (torch.tensor([2.0, 1.0], device=device), 0.1)
}

# === Define Obstacles (Position, Radius) ===
obstacles = [
    (torch.tensor([0.75, 0.0], device=device), 0.2),
    (torch.tensor([0.75, 1.0], device=device), 0.2)
]

# === Motion Planning Model in Pyro ===
def model():
    # 2D latent state variables over time (ensured on GPU)
    X = pyro.sample("X", dist.Normal(torch.zeros(T, 2, device=device), torch.ones(T, 2, device=device)).to_event(2))

    # Enforce 2D SDE dynamics using Pyro factor penalties
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
        for obs, radius in obstacles:
            dist_to_obstacle = torch.norm(X[t] - obs)
            penalty = torch.exp(-((dist_to_obstacle - radius) ** 2) / 0.05)  # Gaussian penalty
            pyro.factor(f"obstacle_{t}_{obs.tolist()}", -penalty * 1e1)

    # Enforce 2D waypoint constraints
    for t, (mean, var) in waypoints.items():
        pyro.factor(f"waypoint_{t}", -1e6 * torch.sum(((X[t] - mean) ** 2) / var))

# === Variational Guide (for Optimization) ===
def guide():
    loc = pyro.param("loc", torch.zeros(T, 2, device=device))
    scale = pyro.param("scale", torch.ones(T, 2, device=device), constraint=dist.constraints.positive)

    pyro.sample("X", dist.Normal(loc, scale).to_event(2))

# === Inference Optimization ===
pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=JitTrace_ELBO())

# === Perform Inference ===
num_iterations = 5000
for step in range(num_iterations):
    loss = svi.step()
    if step % 100 == 0:
        print(f"Iteration {step}: ELBO Loss = {loss:.4f}")

# === Extract Optimized Trajectory ===
optimized_trajectory = pyro.get_param_store()["loc"].detach().cpu().numpy()
print("Optimized Trajectory:", optimized_trajectory)

# Visualization

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.patches import Circle
import numpy as np

# Extract samples

x = optimized_trajectory[:, 0]
y = optimized_trajectory[:, 1]
# Create line segments
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a colormap based on segment index
norm = plt.Normalize(0, len(segments))
colors = plt.cm.viridis(norm(range(len(segments))))  # You can change cmap

# Create a LineCollection
lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
lc.set_array(np.arange(len(segments)))  # Assign color values

# Plot
fig, ax = plt.subplots()
ax.add_collection(lc)

for _, point in waypoints.items():
    point = point[0].detach().cpu().numpy()
    ax.plot(point[0],point[1], marker="o")

for obs, radius in obstacles:
    obs = obs.detach().cpu().numpy()
    ax.add_patch(Circle(obs, radius, color="black"))
ax.autoscale()
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(lc, label="Time Progression")
plt.savefig("results/pyro.png")
plt.close()
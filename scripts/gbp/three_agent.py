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

T = 10
dt = torch.tensor(0.1, device=device)

# Define time-varying system matrices
def A_t(t):
    return torch.tensor([[-0.1 * t, 0], [0, -0.1 * t]], device=device)

def B_t(t):
    return torch.tensor([[0.5 + 0.05 * t, 0], [0, 0.5 + 0.05 * t]], device=device)

# Define agent1 and agent2 trajectories
x_vals1 = torch.linspace(2, 0, T, device=device)
y_vals1 = torch.linspace(0, 1, T, device=device)
agent1_traj = torch.stack((x_vals1, y_vals1), dim=1)

x_vals2 = torch.linspace(0, 2, T, device=device)
y_vals2 = torch.linspace(0.3, 0.3, T, device=device)
agent2_traj = torch.stack((x_vals2, y_vals2), dim=1)

# Start/goal constraints for agent3
waypoints_agent3 = {
    0: (torch.tensor([0.0, 0.0], device=device), 0.1),
    T - 1: (torch.tensor([2.0, 1.0], device=device), 0.1),
}

# Model definition for agent3
def model(agent1_obs, agent2_obs):
    T, D = agent1_obs.shape
    X = pyro.sample("X", dist.Normal(torch.zeros(T, D, device=device), torch.ones(T, D, device=device)).to_event(2))
    safe_distance = torch.tensor(1.0, device=device)
    mask_threshold = torch.tensor(1.0, device=device)

    for t in range(T - 1):
        A = A_t(t)
        B = B_t(t)
        epsilon_t = pyro.sample(f"epsilon_{t}", dist.Normal(torch.zeros(2, device=device), torch.ones(2, device=device)).to_event(1))

        x_next_pred = X[t] + A @ X[t] * dt
        diffusion = B @ epsilon_t * torch.sqrt(dt)
        deviation = (X[t + 1] - x_next_pred - diffusion) ** 2
        pyro.factor(f"sde_constraint_{t}", -0.5 * deviation.sum() / (0.1 ** 2))

    # Collision avoidance penalty
    for i, other_agent_obs in enumerate([agent1_obs, agent2_obs]):
        distance = torch.norm(X - other_agent_obs, dim=-1)
        mask = distance < mask_threshold
        collision_penalty = torch.where(
            (distance < safe_distance) & mask,
            (safe_distance - distance) ** 2,
            torch.zeros_like(distance, device=device)
        )
        pyro.factor(f"collision_penalty_{i}", -collision_penalty.sum() * 1e2)

    # Start/goal constraint penalties
    for t, (mean, var) in waypoints_agent3.items():
        pyro.factor(f"waypoint_{t}", -1e6 * ((X[t] - mean) ** 2).sum() / var)

# Guide for agent3
def guide(agent1_obs, agent2_obs):
    T, D = agent1_obs.shape
    loc = pyro.param("loc", torch.zeros(T, D, device=device))
    scale = pyro.param("scale", torch.ones(T, D, device=device), constraint=pyro.distributions.constraints.positive)
    pyro.sample("X", dist.Normal(loc, scale).to_event(2))

# Inference setup
pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=JitTrace_ELBO())

num_steps = 5000
for step in range(num_steps):
    loss = svi.step(agent1_traj, agent2_traj)
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss}")

inferred_agent3 = pyro.get_param_store()["loc"].detach().cpu().numpy()

# Visualization
x1, y1 = agent1_traj[:, 0].cpu().numpy(), agent1_traj[:, 1].cpu().numpy()
x2, y2 = agent2_traj[:, 0].cpu().numpy(), agent2_traj[:, 1].cpu().numpy()
x3, y3 = inferred_agent3[:, 0], inferred_agent3[:, 1]

fig, ax = plt.subplots()
ax.plot(x1, y1, label="Agent 1", color="blue")
ax.plot(x2, y2, label="Agent 2", color="orange")
ax.plot(x3, y3, label="Agent 3 (Inferred)", color="green")

ax.scatter(*waypoints_agent3[0][0].cpu().numpy(), color="green", marker="o", label="Agent3 Start")
ax.scatter(*waypoints_agent3[T - 1][0].cpu().numpy(), color="red", marker="x", label="Agent3 Goal")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.set_aspect("equal")
plt.savefig("results/three_agents.png")
plt.close()

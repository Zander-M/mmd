"""
Child Class from Gaussian Diffusion Model
Adding Factor Graph inference steps between denoising 
Provides Run Inference Once function
"""
from copy import copy

import einops
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

from mpd.models.diffusion_models import GaussianDiffusionModel
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, ddpm_sample_fn

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

def build_context(model, dataset, input_dict):
    # input_dict is already normalized
    context = None
    if model.context_model is not None:
        context = dict()
        # (normalized) features of variable environments
        if dataset.variable_environment:
            env_normalized = input_dict[f'{dataset.field_key_env}_normalized']
            context['env'] = env_normalized

        # tasks
        task_normalized = input_dict[f'{dataset.field_key_task}_normalized']
        context['tasks'] = task_normalized
    return context

class GaussianDiffusionFactorModel(GaussianDiffusionModel):

    def __init__(self, model=None, variance_schedule='exponential', n_diffusion_steps=100, clip_denoised=True, predict_epsilon=False, loss_type='l2', context_model=None, **kwargs):
        super().__init__(model, variance_schedule, n_diffusion_steps, clip_denoised, predict_epsilon, loss_type, context_model, **kwargs)

    @torch.no_grad()
    def run_inference_one_step(self, 
                               trajs_normalized,
                               t,
                               agent_idx,
                               context=None, hard_conds=None, 
                               n_samples=1, 
                               factor_guide=None,
                               **diffusion_kwargs):
        """
            Run inference for one step for one agent, conditioned on other agent's trajectories
        """
        trajs_normalized = trajs_normalized.clone()
        traj = trajs_normalized[agent_idx]

        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        if context is not None:
            for k, v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Sample from diffusion model
        samples = self.conditional_sample_one_step(
            traj, t, hard_conds, context=context, batch_size=n_samples, **diffusion_kwargs
        )

        # Factor Graph inference 
        if factor_guide is not None:
            # TODO: factor graph inference here!
            print("Factor Guidance Update!")

        # return the last denoising step
        return samples

    @torch.no_grad()
    def conditional_sample_one_step(self, x, t, hard_conds, horizon=None, batch_size=1, **sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        return self.p_sample_one_step(x, t, hard_conds, **sample_kwargs)

    @torch.no_grad()
    def p_sample_one_step(self, x, t, hard_conds, context=None,
                      sample_fn=ddpm_sample_fn,
                      **sample_kwargs):
        """
        Perform a single denoising step at timestep t
        """
        x = x.clone()
        shape = x.shape
        device = self.betas.device
        batch_size = shape[0]

        # apply hard constraints

        x = apply_hard_conditioning(x, hard_conds)
        t = make_timesteps(batch_size, t, device)

        # One step of denoising using the sampling function
        x, values = sample_fn(self, x, hard_conds, context, t, **sample_kwargs)
        x = apply_hard_conditioning(x, hard_conds)
        return x

    def apply_factor_inference(self, x, t_idx, 
                               num_svi_steps=100,
                               step_size=0.01,
                               collision_margin=0.1):
        """
        Use Pyro to run inference over a factor graph that enforces:
        1. Piecewise linearity of trajectories.
        # TODO: add obstacles, add other agents state with variance schedule

        """
        # Detach and allow gradients for optimization
        x = x.detach().clone()
        x.requires_grad = True
    
        batch_size, num_agents, dim = x.shape
    
        # Pyro model
        def model():
            for b in range(batch_size):
                for i in range(num_agents - 1):
                    # Piecewise linearity prior
                    prev = pyro.sample(f"x_{b}_{i}", dist.Normal(x[b, i], 0.1))
                    curr = pyro.sample(f"x_{b}_{i+1}", dist.Normal(prev, 0.1))
    
                    # Collision avoidance
                    for obs in self.map.obstacles:  # Assumes a list of circular obstacles
                        center = torch.tensor(obs['center'], device=x.device)
                        radius = obs['radius']
                        dist_to_obs = (curr - center).norm()
                        pyro.factor(f"obs_factor_{b}_{i}", -torch.relu(radius + collision_margin - dist_to_obs) ** 2)
    
        # Guide
        def guide():
            for b in range(batch_size):
                for i in range(num_agents):
                    loc = pyro.param(f"loc_{b}_{i}", x[b, i].clone())
                    scale = pyro.param(f"scale_{b}_{i}", torch.ones_like(x[b, i]) * 0.1, constraint=pyro.distributions.constraints.positive)
                    pyro.sample(f"x_{b}_{i}", dist.Normal(loc, scale))
    
        # Optimizer and inference object
        pyro.clear_param_store()
        svi = SVI(model, guide, Adam({"lr": step_size}), loss=Trace_ELBO())
    
        # Run SVI
        for _ in range(num_svi_steps):
            svi.step()
    
        # Extract inferred mean trajectory
        for b in range(batch_size):
            for i in range(num_agents):
                x[b, i] = pyro.param(f"loc_{b}_{i}").detach()
    
        return x
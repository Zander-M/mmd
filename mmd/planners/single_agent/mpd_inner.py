import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from typing import List

from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraintNoise
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from mmd.models import TemporalUnet, UNET_DIM_MULTS
from mmd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mmd.trainer import get_dataset, get_model
from mmd.utils.loading import load_params_from_yaml
from mmd.planners.single_agent.single_agent_planner_base import SingleAgentPlanner
from mmd.common.pretty_print import *

class MPDEnd2End(SingleAgentPlanner):
    """
        A class that allows repeated calls to the same model with different inputs(at different sampling step)
        This class keeps track of constraints and feed them to the model only when needed
    """
    def __init__(self,
                 model_id: str,
                 start_state_pos: torch.tensor,
                 goal_state_pos: torch.tensor,
                 use_guide_on_extra_objects_only: bool,
                 start_guide_steps_fraction: float,
                 n_guide_steps: int,
                 n_diffusion_steps_without_noise: int,
                 weight_grad_cost_collision: float,
                 weight_grad_cost_smoothness: float,
                 weight_grad_cost_constraints: float,
                 weight_grad_cost_soft_constraints: float,
                 factor_num_interpolated_points_for_collision: float,
                 trajectory_duration: str,
                 device: str,
                 seed: int,
                 results_dir: str,
                 trained_models_dir: str,
                 n_samples: int,
                 **kwargs
                 ):
        super().__init__()
        # The constraints are stored here. This is a list of ConstraintCost.
        self.constraints = []
        self.weight_grad_cost_constraints = weight_grad_cost_constraints
        self.weight_grad_cost_soft_constraints = weight_grad_cost_soft_constraints

        ####################################
        fix_random_seed(seed)

        device = get_torch_device(device)
        tensor_args = {'device': device, 'dtype': torch.float32}
        ####################################
        print(f'####################################')
        print(f'Initializing Planner with Model -- {model_id}')
        # print(f'Algorithm -- {planner_alg}')
        print('Algorithm -- mmd')
        run_prior_only = False
        run_prior_then_guidance = False
        """if planner_alg == 'mmd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            run_prior_only = True
        else:
            raise NotImplementedError"""
        
        ####################################
        model_dir = os.path.join(trained_models_dir, model_id)
        results_dir = os.path.join(model_dir, 'results_inference', str(seed))
        os.makedirs(results_dir, exist_ok=True)

        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

        ####################################
        # Load dataset with env, robot, task. The TrajectoryDataset type is used here.
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=True,
            obstacle_cutoff_margin=0.05,
            **args,
            tensor_args=tensor_args
        )
        # Extract objects from the dataset.
        dataset = train_subset.dataset
        # Number of support points in the trajectory.
        n_support_points = dataset.n_support_points
        state_dim = dataset.state_dim
        # The environment.
        env = dataset.env      
        # The robot. Contains the dt, the joint limits, etc.
        robot = dataset.robot
        # The task, commonly PlanningTask, is in charge of objects, extra objects, collisions, etc.
        task = dataset.task

        dt = trajectory_duration / n_support_points  # time interval for finite differences

        # set robot's dt
        robot.dt = dt
        ####################################
        # Load prior model
        diffusion_configs = dict(
            variance_schedule=args['variance_schedule'],
            n_diffusion_steps=args['n_diffusion_steps'],
            predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=args['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
        )
        diffusion_model = get_model(
            model_class=args['diffusion_model_class'],
            model=TemporalUnet(**unet_configs),
            tensor_args=tensor_args,
            **diffusion_configs,
            **unet_configs
        )        
        diffusion_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args[
                'use_ema'] else 'model_current_state_dict.pth'),
                       map_location=tensor_args['device'])
        )
        diffusion_model.eval()
        model = diffusion_model
        freeze_torch_model_params(model)
        model = torch.compile(model)
        model.warmup(horizon=n_support_points, device=device)
        ####################################
        # If the args specify a test start and goal, use those.
        if start_state_pos is not None and goal_state_pos is not None:
            print(f'start_state_pos: {start_state_pos}')
            print(f'goal_state_pos: {goal_state_pos}')
        else:
            # Random initial and final positions
            n_tries = 100
            start_state_pos, goal_state_pos = None, None
            for _ in range(n_tries):
                q_free = task.random_coll_free_q(n_samples=2)
                start_state_pos = q_free[0]
                goal_state_pos = q_free[1]

                if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:
                    break

        if start_state_pos is None or goal_state_pos is None:
            raise ValueError(f"No collision free configuration was found\n"
                             f"start_state_pos: {start_state_pos}\n"
                             f"goal_state_pos:  {goal_state_pos}\n")

        print(f'start_state_pos: {start_state_pos}')
        print(f'goal_state_pos: {goal_state_pos}')
        

        ####################################
        # Run motion planning inference

        ########
        # normalize start and goal positions
        hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
        context = None
        print(f'debug: get hard_conds via dataset.get_hard_conditions: {hard_conds}')

        ########
        # Set up the planning costs
        # Cost collisions   
        cost_collision_l = []
        weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
        if use_guide_on_extra_objects_only:
            collision_fields = task.get_collision_fields_extra_objects()
        else:
            collision_fields = task.get_collision_fields()

        print('debug: prepare to constract cost_collision_list')
        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    robot, n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
            weights_grad_cost_l.append(weight_grad_cost_collision)

        # Cost smoothness
        cost_smoothness_l = [
            CostGPTrajectory(
                robot, n_support_points, dt, sigma_gp=1.0,
                tensor_args=tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition
        cost_func_list = [
            *cost_collision_l,
            *cost_smoothness_l,
            # *cost_max_velocity_l,
        ]
        # A `*cost_constraints_l` will be added as "extra cost" and removed after each planning call.

        cost_composite = CostComposite(
            robot, n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=tensor_args
        )
        print('debug: cost_composite list got')

        ########
        # Guiding manager
        guide = GuideManagerTrajectoriesWithVelocity(
            dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=tensor_args,
        )
        print('debug: guide is formulated')

        t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)

        # Keep some variables in the class as members.
        self.start_state_pos = torch.clone(start_state_pos)
        self.goal_state_pos = torch.clone(goal_state_pos)
        self.robot = robot
        self.context = context
        self.run_prior_only = run_prior_only
        self.run_prior_then_guidance = run_prior_then_guidance
        self.n_diffusion_steps_without_noise = n_diffusion_steps_without_noise
        self.hard_conds = hard_conds
        self.model = model
        self.n_support_points = n_support_points
        self.state_dim = state_dim
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.guide = guide
        self.tensor_args = tensor_args
        # Batch-size. How many trajectories to generate at once.
        self.num_samples = n_samples
        # When doing local inference, how many steps to add noise for before denoising again.
        # self.n_local_inference_noising_steps = n_local_inference_noising_steps  # n_local_inference_noising_steps
        # self.n_local_inference_denoising_steps = n_local_inference_denoising_steps
        # Dataset.
        self.dataset = dataset
        # Task, e.g., planning task.
        self.task = task
        # Directories.
        self.results_dir = results_dir

    def __call__(self, start_state_pos, goal_state_pos, constraints_l: List[CostConstraintNoise],
                 *args,
                 **kwargs):
        """
        Call the model with given parameters.
        :param n_samples: Number of trajectories to generate.
        :param start_state_pos: The start state of the robot.
        :param goal_state_pos: The goal state of the robot.
        :param constraints_l: A list of constraints. In this case, interaction among agents
        """
        pass
        
        # Check that the requested start and goal states are similar to the ones stored.
        if not torch.allclose(start_state_pos, self.start_state_pos):
            raise ValueError("The start state is different from the one stored in the planner.")
        if not torch.allclose(goal_state_pos, self.goal_state_pos):
            raise ValueError("The goal state is different from the one stored in the planner.")
        
        # Process the constraints into cost components.
        if constraints_l is not None:
            print("Planning with " + str(len(constraints_l)) + " constraints.")
        else:
            print("Planning without constraints.")

        # NOTE: below were code in self.run_constrained_inference().
        # make constraint_list was moved to end2end_planner.plan()
        self.guide.add_extra_costs(constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in constraints_l])
        trajs_normalized_iters = self.model.run_one_step_inference(
            self.context, self.hard_conds,
            n_samples=n_samples, horizon=self.n_support_points,
            return_chain=True,
            sample_fn=ddpm_sample_fn,
            **self.sample_fn_kwargs,
            n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
        )

    def run_one_step_inference(self, context=None, hard_conds=None, n_samples=1, **diffusion_kwargs):
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)

        #repeat hard conditions and context for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state
        
        if context is not None:
            for k,v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)
        
        #sample from diffusion model
        # NOTE: remove the conditional_sample functions, consider only p_sample_loop for now
        samples, trajs_normalized = self.model.p_sample_loop(hard_conds, context=context,)
        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        # one step sampling -> no chains & no diffsteps
        # trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')
        

        ########
        # if final step, run extra guiding steps without diffusion
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guidance_steps = (self.t_start_guide +
                                               self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guidance_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds = self.hard_conds,
                        guide=self.guide,
                        n_guide_steps=1,
                        unormalize_data=False,
                    )
                    trajs_post_diff_l.append(trajs)
                
                # chain = torch.stack(trajs_post_diff_l, dim=1)
                # chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                # traj_normalized_iters = torch.cat((trajs_normalized_iters, chain))
                traj_normalized_iters = torch.cat((trajs_normalized_iters))
            t_post_diffusion_guide = t_post_diffusion_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')
        
        # Remove the extra cost
        self.guide.reset_extra_costs()

        return trajs_normalized, t_post_diffusion_guide

    def update_constraints(self, constraint_l):
        cost_constraints_l = []
        for c in constraint_l:
            # TODO: what's the difference between soft & not
            c.is_soft = False
            # Clip to range
            c.t_range_l = [(max(0, min(t_range[0], params.horizon - 1)),
                            min(params.horizon - 1, t_range[1]))
                        for t_range in c.t_range_l]
            cost_constraints_l.append(
                CostConstraintNoise(
                    self.robot,
                    self.n_support_points,
                    model_var=c.model_var,
                    q_l=c.get_q_l(),
                    traj_range_l=c.get_t_range_l(),
                    radius_l=c.radius_l,
                    is_soft=c.is_soft,
                    tensor_args=self.tensor_args
                )
            )
        self.guide.add_extra_costs(cost_constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in cost_constraints_l])

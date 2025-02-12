import os
import random
import time
from copy import copy
import einops
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List

from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostConstraint
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from mmd.common.conflicts import Conflict
from mmd.common.constraints import MultiPointConstraintNoise
from mmd.common.experiments import TrialSuccessStatus
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from mmd.common.pretty_print import *
from mmd.config import MMDParams as params
from mmd.common import smooth_trajs, is_multi_agent_start_goal_states_valid, global_pad_paths
from mmd.models.diffusion_models.sample_functions import apply_hard_conditioning, ddpm_sample_fn

def make_timesteps(batch_size, i ,device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

class End2EndPlanning:
    """
    End2End method Based on Diffusion Models
    """
    def __init__(self, single_agent_planner_l,
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 start_consider_collision_step: int = 5,
                 device = None,
                 start_time_l: List[int] = None,
                 reference_robot=None,
                 reference_task=None,
                 **kwargs):
        # Some parameters:
        # self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        self.start_consider_collision_step = start_consider_collision_step
        self.single_agent_planner_l = single_agent_planner_l
        self.n_diffusion_steps = single_agent_planner_l[0].model.n_diffusion_steps
        if self.start_consider_collision_step >= self.n_diffusion_steps:
            raise ValueError(f"start_consider_collision_step: {self.start_consider_collision_step} should be smaller than n_diffusion_steps: {self.n_diffusion_steps}")
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        self.device = device if device else 'cpu'
        if start_time_l is None:
            start_time_l = [0] * self.num_agents
        else:
            self.start_time_l = start_time_l
        # Keep a reference robot for collision checking in a group of robots.
        if reference_robot is None:
            print(CYAN + 'Using the first robot in the low level planner list as the reference robot.' + RESET)
            self.reference_robot = self.single_agent_planner_l[0].robot
        else:
            self.reference_robot = reference_robot
        if reference_task is None:
            print(CYAN + 'Using the first task in the low level planner list as the reference task.' + RESET)
            self.reference_task = self.single_agent_planner_l[0].task
        else:
            self.reference_task = reference_task
        # Check for collisions between robots, and between robots and obstacles, in their start and goal states.
        if not is_multi_agent_start_goal_states_valid(self.reference_robot,
                                                      self.reference_task,
                                                      self.start_state_pos_l,
                                                      self.goal_state_pos_l):
            raise ValueError('Start or goal states are invalid.')
        
  
    def render_paths(self, paths_l: List[torch.Tensor], constraints_l: List[MultiPointConstraintNoise]=None,
                    animation_duration: float = 10.0, output_fpath=None, n_frames=None, plot_trajs=True,
                    show_robot_in_image=True):
        # Render
        planner_visualizer = PlanningVisualizer(
            task = self.reference_task,
        )
        # Add batch dimension to all paths.
        # import pdb; pdb.set_trace()
        # paths_l = [path.unsqueeze(0) for path in paths_l]

        # If animation_duration is None or 0, don't animate and save an image instead.
        if animation_duration is None or animation_duration == 0:
            fig, ax = create_fig_and_axes()
            for agent_id in range(self.num_agents):
                planner_visualizer.render_robot_trajectories(
                    fig=fig,
                    ax=ax,
                    trajs=paths_l[agent_id],
                    start_state=self.start_state_pos_l[agent_id],
                    goal_state=self.goal_state_pos_l[agent_id],
                    colors=[self.agent_color_l[agent_id]],
                    show_robot_in_image=show_robot_in_image
                )
            if output_fpath is None:
                output_fpath = os.path.join(self.results_dir, 'robot-traj.png')
            if not output_fpath.endswith('.png'):
                output_fpath = output_fpath + '.png'
            print(f'Saving image to: file://{os.path.abspath(output_fpath)}')
            plt.axis('off')
            plt.savefig(output_fpath, dpi=100, bbox_inches='tight', pad_inches=0)
            return

        base_file_name = Path(os.path.basename(__file__)).stem
        if output_fpath is None:
            output_fpath = os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif')
        # Render the paths.
        print(f'Rendering paths and saving to: file://{os.path.abspath(output_fpath)}')
        planner_visualizer.animate_multi_robot_trajectories(
            trajs_l=paths_l,
            start_state_l=self.start_state_pos_l,
            goal_state_l=self.goal_state_pos_l,
            plot_trajs=plot_trajs,
            video_filepath=output_fpath,
            n_frames=max((2, paths_l[0].shape[1])) if n_frames is None else n_frames,
            # n_frames=pos_trajs_iters[-1].shape[1],
            anim_time=animation_duration,
            constraints=constraints_l,
            colors=self.agent_color_l
        )

    def plan(self, t_start_guide, n_diffusion_steps_without_noise, 
             sample_fn=ddpm_sample_fn, return_chain=True, warm_start_path_b=None, runtime_limit=1000,):
        """
        Plan a path from start to goal. Do it for one agent at a time.
        shape = (batch_size, horizon, state_dim)
        """
        start_time = time.time()
        # success_status = TrialSuccessStatus.UNKNOWN
        num_agent = len(self.single_agent_planner_l)
        device = self.device
        batch_size = params.n_samples
        # horizon = n_support_point
        shape = (self.num_agents, batch_size, params.horizon, self.single_agent_planner_l[0].state_dim)
        
        if warm_start_path_b is not None:
            x = warm_start_path_b
            print(CYAN, "Using warm start path in p_sample_loop. Steps (negative attempts to recon. t=0)", [self.n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        else:
            x = torch.randn(shape, device=device)
            print(CYAN, "Using random noise in p_sample_loop. Steps (negative attempts to recon. t=0)", [self.n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        #if 't_start_guide' in sample_kwargs:
        #    print(CYAN, "Starting to guide after t <", sample_kwargs['t_start_guide'], RESET)
        print(CYAN, "Starting to guide after t <", t_start_guide, RESET)
        # x = apply_hard_conditioning(x, agent.hard_conds)
        chain = [x] if return_chain else None

        prev_step_trajs = x
        constraint_l = [[] for _ in range(num_agent)]
        trial_success_status_l = [TrialSuccessStatus.UNKNOWN for _ in range(num_agent+1)]
        with TimerCUDA() as timer_inference:
            for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
                print(f'denoising step: {i}')
                t = make_timesteps(batch_size, i, device)
                for j in range(num_agent):
                    agent = self.single_agent_planner_l[j]

                    # context and hard_conds must be normalized
                    hard_conds = copy(agent.hard_conds)
                    context = copy(agent.context)
                    # repeat hard conditions and contexts for n_samples(batch_size)
                    for k, v in hard_conds.items():
                        new_state = einops.repeat(v, 'd -> b d', b=batch_size)
                        hard_conds[k] = new_state
                    
                    if context is not None:
                        for k, v in context.items():
                            context[k] = einops.repeat(v, 'd -> b d', b=batch_size)
                    
                    x[j] = apply_hard_conditioning(x[j], hard_conds)
                    # NOTE: constraint_l: same type with PP's self.create_soft_constraints_from_other_agents_paths(root, agent_id=j)
                    # type(constraint_l[j]) = List[]
                    if int(t[0]) < 0:
                        t = torch.zeros_like(t)
                    model_log_variance = agent.model.posterior_log_variance_clipped.gather(-1, t).reshape(batch_size, *((1,) * (len(x[j].shape) - 1)))
                    model_var = torch.exp(model_log_variance)
                    constraint_l[j] = self.create_soft_constraints_from_other_agents_paths(prev_step_trajs, agent_id=j, step=int(t[0]), model_var=model_var)
                    agent.update_constraints(constraint_l[j])

                    x[j], _ = sample_fn(agent.model, x[j], hard_conds, context, t, guide=agent.guide)
                    x[j] = apply_hard_conditioning(x[j], hard_conds)
                    # remove the added collision constraint(among agents)
                    agent.guide.reset_extra_costs()
                    prev_step_trajs[j] = x[j]
                if return_chain:
                    chain.append(x)
                if time.time() - start_time > runtime_limit:
                    print("Runtime limit reached")
                    trial_success_status_l[j] = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
        t_total_sampling = timer_inference.elapsed
        print(f'sampling: {t_total_sampling:.3f} sec')

        # un-normalize trajectory samples from the models
        # chain.shape = [diffsteps, num_steps(batch_size), horizon, dim] - seems do not need to change
        trajs_normalized_iters = torch.stack(chain, dim=0)

        ####################
        # run extra guiding steps without diffusion
        # TODO
        # NOTE: could consider using LNS as post-diffusion process
        # NOTE: still need to add extra cost before this 
        # with TimerCUDA() as timer_post_model_sample_guide:
        #   trajs = traj_unormalized[-1]
        #   trajs_post_diff_l =[[] for _ in range(num_agent)]
        #   for i in range(num_agent): --- could be reverse: for i in range(n_post_diffusion_guide_steps)
        #       agent = self.single_agent_l[i]
        #       hard_conds = copy(agent.hard_conds)
        #       hard_conds * batch_size, context the same
        #       for j in range(n_post_diffusion_guide_steps):
        #           the input is slightly differen: n_guide_steps=1,unormalize_data=False
        #           trajs[i] = guide_gradient_steps(trajs[i], hard_conds, context, agent.guide, n_guide_steps=1, unormalize_data=False)
        #           trajs_post_diff_l[i].append(trajs)
        #       agent.guide.reset_extra_costs()
        #   trajs_post_diff = torch.stack(trajs_post_diff_l, dim=1)
        #   trajs_post_diff = einops.rearrange(trajs_post_diff, 'num_agent b post_diff_steps h d ->num_agent post_diff, b, h, d')
        #   trajs_normalized_iters = torch.cat((trajs_normalized_iters, trajs_post_diff))
        #   t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
        

        ######################################
        # Extract the best path from the batch
        # self.recent_call_data_l = []
        # the result returned to main
        best_path_l  = []
        # t_total = time.time() - start_time

        for i in range(num_agent):
            agent = self.single_agent_planner_l[i]

            # un-normalize trajectory samples from the models
            # trajs_normalized_iters.shape = (diffstep, n, B, H, q_dim) -> traj_iters (diffstep, B, H, q_dim)
            traj_iters = agent.dataset.unnormalize_trajectories(trajs_normalized_iters[:, i])

            # trajectory of the final diffusion step output
            trajs_final = traj_iters[-1]    # shape = (B, H, q_dim)

            trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = (
                agent.task.get_trajs_collision_and_free(trajs_final, return_indices=True))
            # compute best traj
            idx_best_traj = None  # Index of the best trajectory in the list of all trajectories (trajs_final).
            idx_best_free_traj = None  # Index of the best trajectory in the list of all free trajs (trajs_final_free).
            cost_best_free_traj = None  # Cost of the best trajectory.
            cost_smoothness = None  # Cost of smoothness for all free trajectories.
            cost_path_length = None  # Cost of path length for all free trajectories.
            cost_all = None  # Cost of all factors for all free trajectories. This is a combination of some costs.
            # variance_waypoint_trajs_final_free = None  # Variance of waypoints for all free trajectories.

            if trajs_final_free is not None:
                cost_smoothness = compute_smoothness(trajs_final_free, agent.robot)
                print(f'#{i} agent cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

                cost_path_length = compute_path_length(trajs_final_free, agent.robot)
                print(f'#{i} agent cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

                # Compute best trajectory.
                cost_all = cost_path_length + cost_smoothness
                idx_best_free_traj = torch.argmin(cost_all).item()
                idx_best_traj = trajs_final_free_idxs[idx_best_free_traj]
                cost_best_free_traj = torch.min(cost_all).item()
                print(f'#{i} agent cost best: {cost_best_free_traj:.3f}')
                # variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, agent.robot)
                # even it's stored in planner_output, but it's never called? 
            
            # Smooth the trajectories in trajs_final
            if trajs_final is not None:
                # smooth_trajs(trajs, window_size=10, poly_order=2)
                # trajs, 1. List of trajectories, Each is a tensor of shape (H, q_dim) 2.tensor.shape = (B, H, dim)
                trajs_final = smooth_trajs(trajs_final)
            
            if trajs_final_free_idxs.shape[0]==0:
                best_path_l.append(trajs_final)
                trial_success_status_l[i] = TrialSuccessStatus.FAIL_NO_SOLUTION
                continue

            # check if the runtime limit has been reached
            # if so, return whatever we have
            if time.time() - start_time > runtime_limit:
                print(f"Runtime Limit Reached at agent #{i}")
                trial_success_status_l[num_agent] = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                print('wrong time limit, about to return to main')
                import pdb; pdb.set_trace()
                return best_path_l, 0, trial_success_status_l, []
            # Extract the best path from the batch
            single_agent_best_path_l = [trajs_final[ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch 
                        in enumerate(idx_best_traj)]  
            
            # Global pad before returning.
            # best_path_l: List[torch.Tensor], start_time_l:List[int]
            single_agent_best_path_l = global_pad_paths(single_agent_best_path_l, self.start_time_l)
            single_agent_best_paths = torch.stack(single_agent_best_path_l)
            best_path_l.append(single_agent_best_paths)

        print('finished planning')
        # Check for conflicts
        conflict_l = self.get_conflicts(best_path_l)
        print(RED + 'Conflicts root node:', len(conflict_l), RESET)
        # if every single agent success and the final multi-agent path has no conflict, return success
        if all(item == TrialSuccessStatus.UNKNOWN for item in trial_success_status_l):
            if len(conflict_l) > 0:
                # success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
                trial_success_status_l[num_agent] = TrialSuccessStatus.FAIL_COLLISION_AGENTS
            else:
                # success_status = TrialSuccessStatus.SUCCESS
                trial_success_status_l[num_agent] = TrialSuccessStatus.SUCCESS
        
        
        # best_path, num_ct_expansions, trial_success_status, num_collisions_in_solution in priority_planning.plan
        return best_path_l, [0 for _ in range(num_agent)], trial_success_status_l, len(conflict_l)

    def create_soft_constraints_from_other_agents_paths(self, prev_trajs: torch.Tensor, agent_id: int, step:int, model_var=None,) -> List[MultiPointConstraintNoise]:
        """
        Create soft constraints from the paths of other agents.
        prev_trajs: torch.Tensor. trajectories at t-1 diffusion step for all agents.
        prev_trajs.shape = [num_agent, batch_size, horizon, q_dim]
        ix_best_path_in_batch_l
        """
        if len(prev_trajs) == 0:
            return []
        
        if step < self.start_consider_collision_step:
            return []

        agent_constraint_l = []
        q_l = []
        t_range_l = []
        radius_l = []
        num_agents_in_state = len(prev_trajs)

        ix_best_path_in_batch_l = self.compute_traj_cost(prev_trajs)

        for agent_id_other in range(num_agents_in_state):
            if agent_id_other != agent_id:
                best_path_other_agent = \
                    prev_trajs[agent_id_other][ix_best_path_in_batch_l[agent_id_other]].squeeze(0)
                best_path_other_agent = self.reference_robot.get_position(best_path_other_agent)
                for t_other_agent in range(0, len(best_path_other_agent), 1):
                    t_agent = t_other_agent + self.start_time_l[agent_id_other] - self.start_time_l[agent_id]
                    # The last timestep index for this agent is the lenfth of its path - 1.
                    # If it does not have a path stored, then create constraints for all timesteps
                    # in the path of the other agent (starting from zero).
                    T_agent = len(prev_trajs[agent_id_other][0]) - 1
                    if agent_id >= len(prev_trajs): # final agent
                        T_agent = len(best_path_other_agent) - 1 
                    else:
                        T_agent = len(prev_trajs[agent_id][0]) - 1  
                    
                    if 1 <= t_agent <= T_agent:
                        q_l.append(best_path_other_agent[t_other_agent])
                        t_range_l.append((t_agent, t_agent + 1))
                        radius_l.append(params.vertex_constraint_radius)
        
        if len(q_l) > 0:
            soft_constraint = MultiPointConstraintNoise(q_l=q_l, t_range_l=t_range_l, model_var=model_var)
            soft_constraint.radius_l = radius_l
            soft_constraint.is_soft = True
            # soft_constraint.model_var = model_var
            agent_constraint_l.append(soft_constraint)
        return agent_constraint_l
    
    def compute_traj_cost(self, trajs_final):
        ix_best_path_in_batch_l = []

        for i in range(self.num_agents):
            agent = self.single_agent_planner_l[i]
            idx_best_traj = None  # Index of the best trajectory in the list of all trajectories (trajs_final).
            idx_best_free_traj = None  # Index of the best trajectory in the list of all free trajs (trajs_final_free).
            # cost_best_free_traj = None  # Cost of the best trajectory.
            cost_smoothness = None  # Cost of smoothness for all free trajectories.
            cost_path_length = None  # Cost of path length for all free trajectories.
            cost_all = None  # Cost of all factors for all free trajectories. This is a combination of some costs.
            trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = (
                agent.task.get_trajs_collision_and_free(trajs_final[i], return_indices=True))
            if trajs_final_free is not None:
                cost_smoothness = compute_smoothness(trajs_final_free, agent.robot)
                print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

                cost_path_length = compute_path_length(trajs_final_free, agent.robot)
                print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

                # Compute best trajectory
                cost_all = cost_path_length + cost_smoothness
                idx_best_free_traj = torch.argmin(cost_all).item()
                idx_best_traj = trajs_final_free_idxs[idx_best_free_traj]
                ix_best_path_in_batch_l.append(idx_best_traj)
            else:
                idx_random_choice = random.choice(range(len(trajs_final[i])))
                # idx_best_traj = trajs_final[i][idx_best_free_traj]
                ix_best_path_in_batch_l.append(idx_random_choice)
        return ix_best_path_in_batch_l
        

    def get_conflicts(self, path_l: List[torch.Tensor]) -> List[Conflict]:
        """
        Find conflicts between paths
        TODO:
        path_l.shape = (n, H, q_dim)
        """
        conflicts = []
        return conflicts


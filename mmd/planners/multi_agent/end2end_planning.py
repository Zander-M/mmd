import os
import time
from copy import copy
import einops
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List

from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostConstraint
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from mmd.common.conflicts import Conflict
from mmd.common.constraints import MultiPointConstraint
from mmd.common.experiments import TrialSuccessStatus
from mmd.common.pretty_print import *
from mmd.config import MMDParams as params
from mmd.common import is_multi_agent_start_goal_states_valid, global_pad_paths
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
                 # n_diffusion_steps: int = 50,
                 device = None,
                 start_time_l: List[int] = None,
                 reference_robot=None,
                 reference_task=None,
                 **kwargs):
        # Some parameters:
        # self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        # self.n_diffusion_steps = n_diffusion_steps
        self.single_agent_planner_l = single_agent_planner_l
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        # parameters for diffusion?
        self.device = device if device else 'cpu'
        if start_time_l is None:
            start_time_l = [0] * self.num_agents
        else:
            self.start_time_l = start_time_l
        print(f"debug: {self.start_time_l}")
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
        
  
    def render_path(self):
        # TODO: 
        pass

    def plan(self, t_start_guide, n_diffusion_steps, n_diffusion_steps_without_noise, 
             sample_fn=ddpm_sample_fn, return_chain=True, warm_start_path_b=None, runtime_limit=1000,):
        """
        Plan a path from start to goal. Do it for one agent at a time.
        shape = (batch_size, horizon, state_dim)
        """
        start_time = time.time()
        success_status = TrialSuccessStatus.UNKNOWN
        num_agent = len(self.single_agent_planner_l)
        device = self.device
        batch_size = params.n_samples
        # horizon = n_support_point
        shape = (self.num_agents, batch_size, params.horizon, self.single_agent_planner_l[0].state_dim)
        
        if warm_start_path_b is not None:
            x = warm_start_path_b
            print(CYAN, "Using warm start path in p_sample_loop. Steps (negative attempts to recon. t=0)", [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        else:
            x = torch.randn(shape, device=device)
            print(CYAN, "Using random noise in p_sample_loop. Steps (negative attempts to recon. t=0)", [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        #if 't_start_guide' in sample_kwargs:
        #    print(CYAN, "Starting to guide after t <", sample_kwargs['t_start_guide'], RESET)
        print(CYAN, "Starting to guide after t <", t_start_guide, RESET)
        # x = apply_hard_conditioning(x, agent.hard_conds)
        chain = [x] if return_chain else None

        prev_step_trajs = x
        constraint_l = [[] for _ in range(num_agent)]
        with TimerCUDA() as timer_inference:
            for i in reversed(range(-n_diffusion_steps_without_noise, n_diffusion_steps)):
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
                    # Note sure which should be input if there's no "root"
                    constraint_l[j] = self.create_soft_constraints_from_other_agents_paths(prev_step_trajs, agent_id=j)
                    print("debug: constraint_list respect to other agents is created")
                    agent.update_constraints(constraint_l[j])
                    # NOTE: model_variance is obtained by agent.model.p_mean_variance
                    # this function is called in sample_fn, but the value(variance) is not returned
                    # could return the variance of this step by creating a new sample_fn_return_var
                    print("debug: agent updated with the constraint list")

                    x[j], _ = sample_fn(agent.model, x[j], hard_conds, context, t, guide=agent.guide)
                    x[j] = apply_hard_conditioning(x[j], hard_conds)
                    # remove the added collision constraint(among agents)
                    agent.guide.reset_extra_costs()
                    print("debug: agent.guide.reset_extra_costs(). need to check if the self.single_agent_planner_l[j] is updated")
                    prev_step_trajs[j] = x[i]
                if return_chain:
                    chain.append(x)
                
                if time.time() - start_time > runtime_limit:
                    print("Runtime limit reached")
                    success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT

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

        self.recent_call_data_l = []
        best_path_l = []
        t_total = time.time() - start_time
        # un-normalize trajectory samples from the models
        for i in range(num_agent):
            agent = self.single_agent_planner_l[i]
            traj_iters = agent.dataset.unnormalize_trajectories(chain[i])
            trajs_final = traj_iters[-1]
            trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = (
                agent.task.get_trajs_collision_and_free(trajs_final, return_indices=True))
            # compute best traj
            idx_best_traj = None  # Index of the best trajectory in the list of all trajectories (trajs_final).
            idx_best_free_traj = None  # Index of the best trajectory in the list of all free trajs (trajs_final_free).
            cost_best_free_traj = None  # Cost of the best trajectory.
            cost_smoothness = None  # Cost of smoothness for all free trajectories.
            cost_path_length = None  # Cost of path length for all free trajectories.
            cost_all = None  # Cost of all factors for all free trajectories. This is a combination of some costs.
            variance_waypoint_trajs_final_free = None  # Variance of waypoints for all free trajectories.
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

                variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, agent.robot)
            
            # TODO: we might not need this PlannerOutput class. it's related to updating PP's node
            # personaly i don't think we need this -> it's an end-to-end way, not like PP
            single_agent_output = PlannerOutput()
            single_agent_outputtrajs_iters = trajs_normalized_iters[i]  # chain
            single_agent_output.trajs_final = trajs_final# chain[-1]
            single_agent_output.trajs_final_coll = trajs_final_coll
            single_agent_output.trajs_final_coll_idxs = trajs_final_coll_idxs
            single_agent_output.trajs_final_free = trajs_final_free  # in MPD shape[B,H,D] -> []
            single_agent_output.t_total = t_total
            single_agent_output.trajs_final_free_idxs = trajs_final_free_idxs  # Shape [B]
            single_agent_output.idx_best_traj = idx_best_traj
            single_agent_output.idx_best_free_traj = idx_best_free_traj
            single_agent_output.cost_best_free_traj = cost_best_free_traj
            single_agent_output.cost_smoothness = cost_smoothness
            single_agent_output.cost_path_length = cost_path_length
            single_agent_output.cost_all = cost_all
            single_agent_output.variance_waypoint_trajs_final_free = variance_waypoint_trajs_final_free
            single_agent_output.constraints_l = constraint_l[i]

            # Smooth the trajectories in trajs_final
            if self.recent_call_data.trajs_final is not None:
                self.recent_call_data.trajs_final = smooth_trajs(self.recent_call_data.trajs_final)
            
            if single_agent_output.trajs_final_free_idxs.shape[0]==0:
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                break
            # check if the runtime limit has been reached
            if time.time() - start_time > runtime_limit:
                print("Runtime Limit Reached")
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                return None, 0, success_status, len(conflict_l[i])
            
            # Extract the best path from the batch
            single_agent_best_path_l = [single_agent_output.trajs_final[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch 
                        in enumerate(single_agent_output.idx_best_traj)]
            # TODO: check if we need this plannerOutput
            single_agent_conflict_l = self.get_conflicts(root)
            print(RED + 'Conflicts root node:', len(conflict_l), RESET)
            if success_status == TrialSuccessStatus.UNKNOWN:
                if len(conflict_l) > 0:
                    success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
                #else:
                    #success_status = TrialSuccessStatus.SUCCESS
            
            # Global pad before returning.
            # best_path_l: List[torch.Tensor], start_time_l:List[int]
            single_agent_best_path_l = global_pad_paths(single_agent_best_path_l, agent.start_time_l)
            #TODO: stack the single_agent_best_path_l
            best_path_l.append(single_agent_best_path_l)



        #TODO: check if we need extract the best path from the batch
        #best_path_l = [root.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
        #               enumerate(root.ix_best_path_in_batch_l)]

        # Check for conflicts
        # TODO: check for conflicts, sth like: conflict_l = self.get_conflicts(root)
        # TODO: TOO SLEEPY. CAN"T WORK ANYMORE
        conflict_l = self.get_conflicts(best_path_l)
        print(RED + 'Conflicts root node:', len(conflict_l), RESET)
        if success_status == TrialSuccessStatus.UNKNOWN:
            if len(conflict_l) > 0:
                success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
            else:
                success_status = TrialSuccessStatus.SUCCESS


        return best_path_l, num_ct_expansions_l, trial_success_status_l, num_collisions_in_solution_l
    
    def create_soft_constraints_from_other_agents_paths(self, prev_trajs, agent_id: int) -> List[MultiPointConstraint]:
        if not prev_trajs:
            return []
        
        agent_constraint_l = []
        # might not need this?
        t_range_l = []
        for agent_id_other in range(self.num_agents):
            if agent_id_other != agent_id:
                pass
        
        # if len(q_l)>0:

        return agent_constraint_l

    def get_conflicts(self, path_l) -> List[Conflict]:
        """
        Find conflicts between paths
        TODO: data structure matters
        """
        conflicts = []
        return conflicts#


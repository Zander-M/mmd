import os
import time
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

class End2EndPlanner:
    """
    End2End method Based on Diffusion Models
    """
    def __init__(self, single_agent_planner_l,
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 n_diffusion_steps: int = 50,
                 device = None,
                 start_time_l: List[int] = None,
                 reference_robot=None,
                 reference_task=None,
                 **kwargs):
        # Some parameters:
        # self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        self.n_diffusion_steps = n_diffusion_steps
        self.single_agent_planner_l = single_agent_planner_l
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
        
        # TODO: init each single mpd here
    
    def render_path(self):
        # TODO: 
        pass

    def plan(self, shape, sample_fn=ddpm_sample_fn, return_chain=True, warm_start_path_b=None, runtime_limit=1000, ):
        """
        Plan a path from start to goal. Do it for one agent at a time.
        shape = (batch_size, horizon, state_dim)
        """
        # TODO: source of these params is not set
        success_status = TrialSuccessStatus.UNKNOWN
        device = self.device
        batch_size = shape[0]
        horizon = shape[1]
        # horizon = n_support_point

        if warm_start_path_b is not None:
            x = warm_start_path_b
            print(CYAN, "Using warm start path in p_sample_loop. Steps (negative attempts to recon. t=0)", [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        else:
            x = torch.randn(shape, device=device)
            print(CYAN, "Using random noise in p_sample_loop. Steps (negative attempts to recon. t=0)", [n_diffusion_steps, -n_diffusion_steps_without_noise], RESET)
        if 't_start_guide' in sample_kwargs:
            print(CYAN, "Starting to guide after t <", sample_kwargs['t_start_guide'], RESET)
        x = apply_hard_conditioning(x, agent.hard_conds)
        chain = [x] if return_chain else None

        prev_trajs = chain if return_chain else None
        with TimerCUDA() as timer_inference:
            for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
                # TODO: what's make_timesteps, why do we need it
                t = make_timesteps(batch_size, i, device)
                for j in range(len(self.single_agent_planner_l)):
                    agent = self.single_agent_planner_l[j]
                    cost_constraints_l = []
                    constraint_l = self.create_soft_constraints_from_other_agents_paths(prev_trajs, agent_id=j)
                    agent.update_constraints(constraint_l)
                    # TODO: do we need to store the variance along?
                    x, _ = sample_fn(agent.model, x, agent.hard_conds, agent.context, t, guide=agent.guide)
                    x = apply_hard_conditioning(x, agent.hard_conds)
                    # remove the added collision constraint(among agents)
                    agent.guide.reset_extra_costs()
                    prev_trajs[j] = x[-1]

        t_total = timer_inference.elapsed
        print(f'sampling: {t_total:.3f} sec')

        #TODO: check if we need extract the best path from the batch
        #best_path_l = [root.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
        #               enumerate(root.ix_best_path_in_batch_l)]

        # Check for conflicts
        # TODO: check if we need check for conflicts
        # origin: conflict_l = self.get_conflicts(root)
        conflict_l = self.get_conflicts(best_path_l)
        print(RED + 'Conflicts root node:', len(conflict_l), RESET)
        if success_status == TrialSuccessStatus.UNKNOWN:
            if len(conflict_l) > 0:
                success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
            else:
                success_status = TrialSuccessStatus.SUCCESS

        # Global pad before returning.
        # best_path_l: List[torch.Tensor], start_time_l:List[int]
        best_path_l = global_pad_paths(best_path_l, self.start_time_l)

        return best_path_l, t_total, success_status, len(conflict_l)
    
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
        return conflicts


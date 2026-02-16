"""
HARL wrapper for Multi-Objective JUSTICE Environment.

This wrapper adapts the JusticeEnvironmentMO (MOMAland MOParallelEnv) to work with
the HARL framework. It applies MOMAland's LinearizeReward and NormalizeReward
wrappers to scalarize multi-objective rewards for training.
"""

from thesis_rl.envs.justice_environment_moma import JusticeEnvironmentMOMA
from thesis_rl.args import EnvArgs

from momaland.utils.parallel_wrappers import (
    LinearizeReward,
    NormalizeReward,
)

import numpy as np
from gymnasium.spaces import Box, Discrete
import copy


class HarlJusticeMOMARLEnvironment:
    """
    HARL-compatible wrapper for multi-objective Justice environment.
    
    Applies MOMAland wrappers to linearize multi-objective rewards into scalar
    rewards that HARL runners can process.
    
    Args:
        args: Dictionary containing environment configuration including:
            - weights: List of floats for reward linearization
            - rewards: List of objective names
            - num_agents, ensables, state_type, num_actions, action_change
            - normalize_rewards: Whether to normalize rewards (default: True)
    """
    
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        
        # Extract weights for linearization
        self.weights = np.array(args.get('weights', [0.5, 0.5]), dtype=np.float32)
        self.normalize_rewards = args.get('normalize_rewards', True)
        
        # Create EnvArgs for the base environment
        env_args = EnvArgs(**{k: v for k, v in self.args.items() 
                             if k in EnvArgs.__dataclass_fields__})
        
        # Create the base MOMA environment
        self.base_env = JusticeEnvironmentMOMA(env_args)
        
        # Apply MOMAland wrappers
        self.env = self._apply_momaland_wrappers(self.base_env)
        
        self.n_agents = len(self.env.possible_agents)
        self.agents = self.env.possible_agents
        
        self.share_observation_space = self.repeat(
            self.base_env.state_space()
        )
        
        self.observation_space = self.repeat(
            self.base_env.observation_space(None)
        )
        
        self.action_space = self.repeat(self.base_env.action_space(None))
        self._seed = 0
    
    def _apply_momaland_wrappers(self, env):
        """Apply NormalizeReward and LinearizeReward wrappers."""
        wrapped_env = env
        
        # Normalize each reward dimension per agent
        if self.normalize_rewards:
            for agent in wrapped_env.possible_agents:
                num_objectives = wrapped_env.reward_space(agent).shape[0]
                for idx in range(num_objectives):
                    wrapped_env = NormalizeReward(wrapped_env, agent, idx)
        
        # Linearize rewards with provided weights
        weights_dict = {agent: self.weights for agent in env.possible_agents}
        wrapped_env = LinearizeReward(wrapped_env, weights_dict)
        
        return wrapped_env
    
    def step(self, actions):
        """Execute one step in the environment."""
        obs, rewards, term, trunc, infos = self.env.step(self.wrap(actions.flatten()))
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        state = self.base_env.get_state()
        
        # Rewards are now scalar (linearized) for HARL compatibility
        rewards_list = [[rewards[agent]] for agent in self.agents]
        
        return (
            self.unwrap(obs),
            state,
            rewards_list,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_avail_actions(),
        )
    
    def reset(self):
        """Reset the environment."""
        obs, infos = self.env.reset()
        state = self.base_env.get_state()
        return self.unwrap(obs), state, self.get_avail_actions()

    def get_avail_actions(self):
        """Get available actions for all agents."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
       
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.base_env.action_mask[self.base_env.possible_agents[agent_id]]

    def seed(self, seed):
        """Set the random seed."""
        self._seed = seed

    def render(self):
        """Render the environment."""
        self.base_env.render()

    def close(self):
        """Close the environment."""
        pass

    def wrap(self, l):
        """Convert list to dict keyed by agent names."""
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        """Convert dict to list."""
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        """Repeat single value for all agents."""
        return [a for _ in range(self.n_agents)]
    
    def get_num_objectives(self):
        """Return the number of objectives."""
        return self.base_env.num_objectives
    
    def get_reward_space(self):
        """Return the reward space of the base (unwrapped) environment."""
        return self.base_env.reward_space(self.agents[0])

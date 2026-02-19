"""
HARL wrapper for JUSTICE Environment.

This wrapper adapts the JusticeEnvironment (PettingZoo ParallelEnv) to work with
the HARL (Heterogeneous-Agent Reinforcement Learning) framework. It follows the
HARL environment interface pattern similar to PettingZooMPEEnv.

"""

from thesis_rl.envs.justice_environment import JusticeEnvironment
from thesis_rl.args import EnvArgs

import numpy as np
from gymnasium.spaces import Box, Discrete
import copy

class HarlJusticeEnvironment:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        env_args = EnvArgs(**self.args)
        self.env = JusticeEnvironment(env_args)
        self.n_agents = len(self.env.possible_agents)
        self.agents = self.env.possible_agents
        
        self.share_observation_space = self.repeat(
            self.env.state_space()
        )
        
        self.observation_space = self.repeat(
            self.env.observation_space(None)
        )
        
        # Action space: discrete actions from 0-10 (representing 0.0 to 1.0 emissions control)
        self.action_space = self.repeat(self.env.action_space(None))
        self._seed = 0
    
    def step(self, actions):
        # actions shape: (n_agents, 2) for MultiDiscrete([num_actions, num_actions])
        # Each agent has [emission_control_action, savings_rate_action]
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        obs, rewards, term, trunc, infos = self.env.step(action_dict)
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        state = self.env.get_state()
        rewards = [[rewards[agent]] for agent in self.agents]
        return (
            self.unwrap(obs),
            state,
            rewards,
            self.unwrap(dones),
            self.unwrap(infos),
            self.get_avail_actions(),
        )
    
    def reset(self):
        obs, infos = self.env.reset()
        state = self.env.get_state()
        return self.unwrap(obs), state, self.get_avail_actions()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
       
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.action_mask[self.env.possible_agents[agent_id]]

    def seed(self, seed):
        """Set the random seed"""
        self._seed = seed

    def render(self):
        """Render the environment"""
        self.env.render()

    def close(self):
        """Close the environment"""
        self.env.close()

    def wrap(self, l):
        """Convert list to dict keyed by agent names"""
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        """Convert dict to list"""
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        """Repeat single value for all agents"""
        return [a for _ in range(self.n_agents)]
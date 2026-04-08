import time

import numpy as np
from thesis_rl.HARL.harl.common.base_logger import BaseLogger
import wandb
import random

class HarlJusticeLogger(BaseLogger):
    
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(HarlJusticeLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        # shape: (n_rollout_threads, num_agents, num_objectives)
        self.num_objectives = 2 if env_args.get('reward', 'other') == 'multi_objective' else 1
        self.objective_names = env_args.get('rewards', ['objective_0', 'objective_1'])[:self.num_objectives]
        self.env_agent_rewards = np.zeros((
            self.algo_args["train"]["n_rollout_threads"], self.num_agents, self.num_objectives
            ))
        self.done_env_agent_rewards = []  
    
    def per_step(self, data):
        """Process data per step, including vector rewards from infos."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        
        # Extract vector rewards from infos for logging
        # infos shape: (n_threads,) where each element is a list of agent info dicts
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if infos[t] is not None:
                for agent_idx, info in enumerate(infos[t]):
                    if isinstance(info, dict):
                        vec_reward = info.get('rewards', None)
                        if isinstance(vec_reward, np.ndarray) and len(vec_reward) == self.num_objectives:
                            self.env_agent_rewards[t, agent_idx, :] += vec_reward
                    
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0
                
                self.done_env_agent_rewards.append(self.env_agent_rewards[t, :, :].copy())
                self.env_agent_rewards[t, :, :] = np.zeros((self.num_agents, self.num_objectives))

        
    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args['exp_name'],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        
        if critic_buffer.rewards.ndim == 4:
            # if critic buffer has shape (n_rollout_threads, episode_length, num_agents, 1)
            for agent_id in range(self.num_agents):
                actor_train_infos[agent_id][
                    "average_step_rewards"
                ] = critic_buffer.rewards[:, :, agent_id].mean()
                
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )
        wandb.log({
            "average_step_rewards": critic_buffer.get_mean_rewards(),
            "total_num_steps": self.total_num_steps,
        })
        
        if critic_buffer.rewards.ndim == 4:
            # if critic buffer has shape (n_rollout_threads, episode_length, num_agents, 1)
            for agent_id in range(self.num_agents):
                wandb.log({
                    f"reward_agent{agent_id}": critic_buffer.rewards[:, :, agent_id].sum(axis=0).mean()
                })

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            # env_agent_rewards shape: (num_episodes, num_agents, num_objectives)
            env_agent_rewards = np.array(self.done_env_agent_rewards)
            # Mean across episodes and agents -> (num_objectives,)
            mean_vec_return = env_agent_rewards.mean(axis=(0, 1))
            
            print(
                "Some episodes done, average episode reward is {}.".format(
                    aver_episode_rewards
                )
            )
            # Can be changed to print per agent rewards if needed
            print(
                "Average vector return: {} \n".format(
                    mean_vec_return
                )
            )
            
            # Log scalar metrics for wandb
            log_dict = {
                "aver_episode_rewards": aver_episode_rewards,
            }
            # Log each objective separately for hypervolume computation
            for obj_idx in range(self.num_objectives):
                log_dict[f"{obj_idx}"] = mean_vec_return[obj_idx]
            
            wandb.log(log_dict)
                
            self.done_episodes_rewards = []
            self.done_env_agent_rewards = []
    
    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                
                wandb.log({
                    f"agent{agent_id}/{k}": v
                })
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            
            wandb.log({
                critic_k: v
            })
    
    def eval_init(self):
        """Initialize evaluation tracking including per-objective vector rewards."""
        super().eval_init()
        n_eval_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
        # Per-thread, per-step vector rewards: list of lists
        self.eval_one_episode_vec_rewards = [[] for _ in range(n_eval_threads)]
        # Completed episode vector returns per thread
        self.eval_episode_vec_returns = [[] for _ in range(n_eval_threads)]

    def eval_per_step(self, eval_data):
        """Track per-step vector rewards from eval infos."""
        super().eval_per_step(eval_data)
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_infos[eval_i] is not None:
                # Average vector reward across agents for this thread/step
                step_vec = np.zeros(self.num_objectives)
                count = 0
                for info in eval_infos[eval_i]:
                    if isinstance(info, dict):
                        vec_reward = info.get('rewards', None)
                        if isinstance(vec_reward, np.ndarray) and len(vec_reward) == self.num_objectives:
                            step_vec += vec_reward
                            count += 1
                if count > 0:
                    step_vec /= count
                self.eval_one_episode_vec_rewards[eval_i].append(step_vec)

    def eval_thread_done(self, tid):
        """Accumulate vector returns when an eval episode finishes."""
        super().eval_thread_done(tid)
        if self.eval_one_episode_vec_rewards[tid]:
            episode_vec_return = np.sum(self.eval_one_episode_vec_rewards[tid], axis=0)
            self.eval_episode_vec_returns[tid].append(episode_vec_return)
        self.eval_one_episode_vec_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information including per-objective unnormalized rewards."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))

        log_dict = {
            "eval/eval_average_episode_rewards": eval_avg_rew,
        }

        # Log per-objective unnormalized rewards
        all_vec_returns = [vr for thread_vrs in self.eval_episode_vec_returns for vr in thread_vrs]
        if len(all_vec_returns) > 0:
            vec_returns = np.array(all_vec_returns)  # (num_episodes, num_objectives)
            mean_vec = vec_returns.mean(axis=0)
            max_vec = vec_returns.max(axis=0)

            for obj_idx, obj_name in enumerate(self.objective_names):
                log_dict[f"eval/{obj_idx}_mean"] = mean_vec[obj_idx]
            
            # Log per-episode vector returns as a wandb.Table for reliable retrieval
            columns = [f"obj_{i}" for i in range(self.num_objectives)]
            vec_table = wandb.Table(columns=columns, data=vec_returns.tolist())
            log_dict["eval/objective_vector"] = vec_table

            print("Eval per-objective mean returns: {}\n".format(
                {name: mean_vec[i] for i, name in enumerate(self.objective_names)}
            ))

        wandb.log(log_dict)
        
    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                # self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)
                wandb.log({
                    k: np.mean(v)
                })
    
    def get_task_name(self):
        return "harl_justice"
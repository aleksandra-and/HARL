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
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            # self.writter.add_scalars(
            #     "train_episode_rewards",
            #     {"aver_rewards": aver_episode_rewards},
            #     self.total_num_steps,
            # )
            wandb.log({
                "aver_episode_rewards": aver_episode_rewards
            })
                
            self.done_episodes_rewards = []
    
    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                # self.writter.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)
                wandb.log({
                    f"agent{agent_id}/{k}": v
                })
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            # self.writter.add_scalars(critic_k, {critic_k: v}, self.total_num_steps)
            wandb.log({
                critic_k: v
            })
    
    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval/eval_average_episode_rewards": self.eval_episode_rewards,
            "eval/eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        # self.log_file.write(
        #     ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        # )
        # self.log_file.flush()
        wandb.log({
            "eval/eval_average_episode_rewards": eval_avg_rew
        })
        
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
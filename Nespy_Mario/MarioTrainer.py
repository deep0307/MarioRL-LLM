import os
from collections import deque

import gym
from gym_utils import load_smb_env, SMB

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Baseline reward - this function determines the reward at each step by calculating Mario’s velocity (positive points while moving right, negative points while moving left, zero while standing still),
# plus a penalty for every frame that passes to encourage movement, and a penalty if Mario dies for any reason.

# Human generated reward - this reward function rewards Mario for increasing his in-game score by defeating enemies, grabbing coins, and collecting power-ups.
# Additionally, a sizable reward is added if he collects the flag (or defeats Bowser) at the end of the level to encourage him to successfully beat the stage.
class HumanReward(gym.Wrapper):
    def __init__(self, env):
        super(HumanReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if terminated or truncated:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, terminated, truncated, info
    
# LLM generated rewards (No edits)
# This reward function encourages the agent to complete the level as quickly as possible by rewarding it for decreasing the remaining time.
class TimeReward(gym.Wrapper):
    def __init__(self, env):
        super(TimeReward, self).__init__(env)
        self._current_time = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        time_reward = info['time'] - self._current_time
        self._current_time = info['time']
        reward += time_reward
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to collect coins by rewarding it for increasing the coin count.
class CoinReward(gym.Wrapper):
    def __init__(self, env):
        super(CoinReward, self).__init__(env)
        self._current_coins = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        coin_reward = info['coins'] - self._current_coins
        self._current_coins = info['coins']
        reward += coin_reward * 20
        if terminated or truncated and self._current_coins == 0:
            reward -= 10
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to stay high up on the screen, which can be beneficial for avoiding enemies and obstacles.
class HeightReward(gym.Wrapper):
    def __init__(self, env):
        super(HeightReward, self).__init__(env)
        self._prev_y_pos = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        height_reward = self._prev_y_pos - info['y_pos']
        self._prev_y_pos = info['y_pos']
        reward += height_reward
        return state, reward, terminated, truncated, info
    
# This reward function encourages the agent to move as far as possible to the right, which is the primary objective in Super Mario Bros.
class DistanceReward(gym.Wrapper):
    def __init__(self, env):
        super(DistanceReward, self).__init__(env)
        self._prev_x_pos = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        distance_reward = info['x_pos'] - self._prev_x_pos
        self._prev_x_pos = info['x_pos']
        reward += distance_reward
        return state, reward, terminated, truncated, info

# This reward function rewards the agent for gaining power-ups (e.g., going from small to tall, or from tall to fireball) by increasing its status.
class PowerupReward(gym.Wrapper):
    def __init__(self, env):
        super(PowerupReward, self).__init__(env)
        self._prev_status = env._player_status

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if info['status'] > self._prev_status:
            reward += 50  # Adjust the reward value as needed
        self._prev_status = info['status']
        return state, reward, terminated, truncated, info

# This reward function rewards the agent for killing enemies by tracking the enemy types on the screen and rewarding the agent when the sum of enemy types decreases.
class EnemyKillReward(gym.Wrapper):
    def __init__(self, env):
        super(EnemyKillReward, self).__init__(env)
        self._prev_enemies = None

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        enemies = [self.env.ram[addr] for addr in self.env._ENEMY_TYPE_ADDRESSES]
        if self._prev_enemies is not None:
            enemy_diff = sum(self._prev_enemies) - sum(enemies)
            reward += enemy_diff * 10  # Adjust the reward value as needed
        self._prev_enemies = enemies
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to explore new areas of the level by rewarding it for visiting new (x, y) positions on the screen.
class ExplorationReward(gym.Wrapper):
    def __init__(self, env):
        super(ExplorationReward, self).__init__(env)
        self._visited = set()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        x_pos, y_pos = info['x_pos'], info['y_pos']
        pos = (x_pos, y_pos)
        if pos not in self._visited:
            self._visited.add(pos)
            reward += 1  # Adjust the reward value as needed
        return state, reward, terminated, truncated, info

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, starting_steps=0, verbose=1, prev_stats_dict=None):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.starting_steps = starting_steps

        self.powers_up_list = [0] * 5

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if self.locals['infos'][i]['status'] != 'small':
                self.powers_up_list[i] = 1

            if done:
                print('Episode end', i)
                self.logger.record('rollout/ep_score', self.locals['infos'][i]['score'], exclude='stdout')
                self.logger.record('rollout/num_coins', self.locals['infos'][i]['coins'], exclude='stdout')
                self.logger.record('rollout/flag_get', 1 if self.locals['infos'][i]['flag_get'] else 0, exclude='stdout')
                self.logger.record('rollout/end_x_pos', self.locals['infos'][i]['x_pos'], exclude='stdout')
                self.logger.record('rollout/powers_up', self.powers_up_list[i], exclude='stdout')
                self.powers_up_list[i] = 0
        
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls + int(self.starting_steps)))
            self.model.save(model_path)

        return True

def make_env(render_mode: str = None):
    env = load_smb_env(render_mode=render_mode)
    # Modify the reward function if needed
    env = CoinReward(env)
    return env
    
if __name__ == '__main__':
    CHECKPOINT_DIR = 'checkpoints/coin_reward_2/'
    LOG_DIR = './logs/'
    TOTAL_TIMESTEPS = 3e6
    NUM_CPU = 5

    # # Create the vectorized environment
    vec_env = make_vec_env(make_env, n_envs=NUM_CPU, vec_env_cls=SubprocVecEnv)

    # # Create the callback for training and logging
    callback = TrainAndLoggingCallback(check_freq=25000, save_path=CHECKPOINT_DIR)

    # # Train the AI model, this is where the AI model starts to learn
    model = PPO('MlpPolicy', vec_env, verbose=0, tensorboard_log=LOG_DIR, device='cpu', learning_rate=0.0002, batch_size=256, n_steps=512)
    # model = PPO.load('MarioRL/Human_Reward', env=vec_env, verbose=0, tensorboard_log=LOG_DIR, device='cpu', learning_rate=0.0002, batch_size=256, n_steps=512)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=callback)

    # # Save the AI model
    model.save('MarioRL/Coin_Reward')
    del model

    # Load the AI model
    # eval_env = make_env(render_mode='human')
    # model = PPO.load('MarioRL_Exploration_Reward')

    # Create the SMB wrapper
    # smb = SMB(eval_env, model)
    # smb.play(episodes=10)

    
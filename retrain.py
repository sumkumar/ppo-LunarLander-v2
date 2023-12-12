import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = make_vec_env('LunarLander-v2', n_envs=32)

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters


env_id = "LunarLander-v2"
eval_env = gym.make(env_id)
model_name = "C:/Users/sum/Documents/Worskpace/RL/DeepRL-HF-course/unit 1/ppo-LunarLander-v2-sum"
model = PPO.load(model_name+'.zip', eval_env)

model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v2-sum-v2"
model.save(model_name)

eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
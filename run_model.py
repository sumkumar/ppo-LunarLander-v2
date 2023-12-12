import gym
from huggingface_hub import notebook_login
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env



env_id = "LunarLander-v2"
eval_env = gym.make(env_id)
model_name = "C:/Users/sum/Documents/Worskpace/RL/DeepRL-HF-course/unit 1/ppo-LunarLander-v2-sum-v2"
model = PPO.load(model_name+'.zip', eval_env)


mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

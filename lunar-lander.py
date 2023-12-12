import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = make_vec_env('LunarLander-v2', n_envs=32)

# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    learning_rate = 0.0001,
    n_steps = 1024,
    batch_size = 128,
    n_epochs = 16,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

model.learn(total_timesteps=10000000)
# Save the model
model_name = "ppo-LunarLander-v2-sum"
model.save(model_name)

eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
import gym
from huggingface_hub import notebook_login
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env



env_id = "LunarLander-v2"
eval_env = gym.make(env_id)
model_name = "C:/Users/sum/Documents/Worskpace/RL/DeepRL-HF-course/unit 1/ppo-LunarLander-v2-sum"
model = PPO.load(model_name+'.zip', eval_env)


mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model
               model_architecture="PPO", # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=DummyVecEnv([lambda: gym.make(env_id)]), # Evaluation Environment
               repo_id="sumkumar/ppo-LunarLander-v2-sum", # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message="Updated model")



import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
import optuna
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import AgrikaTractorFleetEnv

class TrainingCallback(BaseCallback):
    """Custom callback to track training metrics and detect convergence"""
    def __init__(self, log_dir, window_size=100, reward_threshold=3000, std_threshold=100):
        super().__init__(verbose=0)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.raw_rewards = []
        self.episode_lengths = []
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            episode_reward = self.locals['rewards'][-1]
            raw_reward = self.locals['infos'][-1].get('raw_reward', episode_reward)
            episode_length = self.locals['infos'][-1].get('day', 60)
            self.episode_rewards.append(episode_reward)
            self.raw_rewards.append(raw_reward)
            self.episode_lengths.append(episode_length)
            if len(self.raw_rewards) >= self.window_size:
                mean_raw_reward = np.mean(self.raw_rewards[-self.window_size:])
                std_raw_reward = np.std(self.raw_rewards[-self.window_size:])
                with open(f"{self.log_dir}/training_metrics.txt", "a") as f:
                    f.write(f"Episode {len(self.episode_rewards)}: Mean Raw Reward = {mean_raw_reward:.2f}, Std = {std_raw_reward:.2f}, Length = {episode_length}\n")
                if mean_raw_reward > self.reward_threshold and std_raw_reward < self.std_threshold:
                    print(f"Converged at episode {len(self.episode_rewards)}: Mean Raw Reward = {mean_raw_reward:.2f}, Std = {std_raw_reward:.2f}")
                    return False  # Stop training
        return True

    def _on_training_end(self):
        np.savetxt(f"{self.log_dir}/raw_rewards.csv", self.raw_rewards, delimiter=",")

def plot_learning_curve(log_dir, title, save_path):
    """Plot learning curve from training logs"""
    try:
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"No data found in {log_dir}/monitor.csv")
            return
        x, y = ts2xy(results, 'timesteps')
        raw_y = (y + 1) / 2 * (500 - (-600)) + (-600)  # Convert normalized to raw
        plt.figure(figsize=(10, 6))
        plt.plot(x, raw_y, label="PPO Raw Reward")
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Raw Reward')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {save_path}")
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

def train_ppo_optimized(trial=None):
    """Train PPO with optimized hyperparameters"""
    os.makedirs("models/pg/ppo", exist_ok=True)
    log_dir = f"logs/ppo/trial_{trial.number}" if trial else "logs/ppo"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting PPO Training (Trial {trial.number if trial else 'Default'})")
    
    env = AgrikaTractorFleetEnv(normalize_rewards=True)
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4]) if trial else 0.0003,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": trial.suggest_categorical("n_epochs", [10, 15, 20]) if trial else 15,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2]) if trial else 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": trial.suggest_categorical("ent_coef", [0.01, 0.05, 0.1]) if trial else 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "policy_kwargs": dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        ),
        "tensorboard_log": log_dir,
        "verbose": 1,
        "device": "auto"
    }
    
    model = PPO(**model_params)
    
    eval_env = Monitor(AgrikaTractorFleetEnv(normalize_rewards=True), filename=f"{log_dir}/eval_monitor.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    training_callback = TrainingCallback(log_dir=log_dir)
    
    print("Training PPO for 500,000 timesteps...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, training_callback],
        progress_bar=True,
        tb_log_name=f"ppo_trial_{trial.number}" if trial else "ppo_optimized"
    )
    
    model.save(f"{log_dir}/final_ppo_model")
    
    plot_learning_curve(log_dir, f"PPO Learning Curve (Trial {trial.number if trial else 'Default'})", 
                        f"{log_dir}/ppo_learning_curve.png")
    
    print(f"✅ PPO training completed for trial {trial.number if trial else 'Default'}!")
    env.close()
    eval_env.close()
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, return_episode_rewards=True)
    raw_rewards = [(r + 1) / 2 * (500 - (-600)) + (-600) for r in mean_reward]
    return np.mean(raw_rewards) - np.std(raw_rewards)  # Objective for optuna

def tune_ppo():
    """Run hyperparameter tuning with optuna"""
    study = optuna.create_study(direction="maximize")
    study.optimize(train_ppo_optimized, n_trials=3)  # Reduced to 3 trials
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    return study

if __name__ == "__main__":
    study = tune_ppo()
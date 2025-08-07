# ============================================================================
# FILE: training/ppo_training.py
# ============================================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import AgrikaTractorFleetEnv


class TrainingCallback(BaseCallback):
    """Custom callback to track training metrics and detect convergence"""
    def __init__(self, log_dir, window_size=100, reward_threshold=3000, std_threshold=100):
        super().__init__()
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        if self.locals.get('dones', False).any():
            episode_reward = self.locals['rewards'][-1]
            episode_length = self.locals['infos'][-1].get('episode', {}).get('l', 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if len(self.episode_rewards) >= self.window_size:
                mean_reward = np.mean(self.episode_rewards[-self.window_size:])
                std_reward = np.std(self.episode_rewards[-self.window_size:])
                with open(f"{self.log_dir}/training_metrics.txt", "a") as f:
                    f.write(f"Episode {len(self.episode_rewards)}: Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}, Length = {episode_length}\n")
                if mean_reward > self.reward_threshold and std_reward < self.std_threshold:
                    print(f"Converged at episode {len(self.episode_rewards)}: Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
                    return False  # Stop training on convergence
        return True

def train_ppo_optimized():
    """Train PPO with optimized hyperparameters"""
    # Create directories
    os.makedirs("models/pg/ppo", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    
    print("🚀 Starting PPO Training with Optimized Hyperparameters")
    
    # Create monitored environment
    env = AgrikaTractorFleetEnv()
    env = Monitor(env, filename="logs/ppo/monitor.csv")
    
    # PPO with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log="logs/ppo",
        verbose=1,
        device="auto"
    )
    
    # Setup evaluation
    eval_env = Monitor(AgrikaTractorFleetEnv(), filename="logs/ppo/eval_monitor.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/pg/ppo/",
        log_path="logs/ppo/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Setup training callback for convergence
    training_callback = TrainingCallback(log_dir="logs/ppo")
    
    # Train the model
    print("Training PPO for 500,000 timesteps...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, training_callback],
        progress_bar=True,
        tb_log_name="ppo_optimized"
    )
    
    # Save final model
    model.save("models/pg/ppo/final_ppo_model")
    
    # Generate learning curve
    plot_learning_curve("logs/ppo", "PPO Learning Curve", "models/pg/ppo/ppo_learning_curve.png")
    
    print("✅ PPO training completed!")
    env.close()
    eval_env.close()
    return model

def plot_learning_curve(log_dir, title, save_path):
    """Plot learning curve from training logs"""
    try:
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"No data found in {log_dir}/monitor.csv")
            return
        x, y = ts2xy(results, 'timesteps')
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="PPO Reward")
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Reward')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {save_path}")
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

if __name__ == "__main__":
    model = train_ppo_optimized()
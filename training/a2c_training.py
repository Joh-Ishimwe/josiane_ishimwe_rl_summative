# ============================================================================
# FILE: training/a2c_training.py
# ============================================================================

import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
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
        if self.locals.get('dones', False):
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

def train_a2c_optimized():
    """Train A2C with optimized hyperparameters"""
    # Create directories
    os.makedirs("models/pg/a2c", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)
    
    print("🚀 Starting A2C Training with Optimized Hyperparameters")
    
    # Create monitored environment
    env = AgrikaTractorFleetEnv()
    env = Monitor(env, filename="logs/a2c/monitor.csv")
    
    # A2C with optimized hyperparameters
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0007,
        n_steps=10,  # Increased for better updates
        gamma=0.99,
        gae_lambda=0.95,  # Adjusted for better advantage estimation
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        tensorboard_log="logs/a2c",
        verbose=1,
        device="auto"
    )
    
    # Setup evaluation
    eval_env = Monitor(AgrikaTractorFleetEnv(), filename="logs/a2c/eval_monitor.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/pg/a2c/",
        log_path="logs/a2c/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Setup training callback for convergence
    training_callback = TrainingCallback(log_dir="logs/a2c")
    
    # Train the model
    print("Training A2C for 500,000 timesteps...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, training_callback],
        progress_bar=True,
        tb_log_name="a2c_optimized"
    )
    
    # Save final model
    model.save("models/pg/a2c/final_a2c_model")
    
    # Generate learning curve
    plot_learning_curve("logs/a2c", "A2C Learning Curve", "models/pg/a2c/a2c_learning_curve.png")
    
    print("✅ A2C training completed!")
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
        plt.plot(x, y, label="A2C Reward")
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
    train_a2c_optimized()
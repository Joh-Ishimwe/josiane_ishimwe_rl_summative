import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO  # Added PPO for commented section
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import EnhancedAgrikaTractorFleetEnv  # Updated to Enhanced environment

class TrainingCallback(BaseCallback):
    """Custom callback to track training metrics and detect convergence"""
    def __init__(self, log_dir, window_size=100, reward_threshold=200, std_threshold=50):
        super().__init__(verbose=1)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.raw_rewards = []
        self.episode_lengths = []
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_count += 1
            # Get episode info from the environment
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.raw_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Check convergence
            if len(self.raw_rewards) >= self.window_size:
                mean_raw_reward = np.mean(self.raw_rewards[-self.window_size:])
                std_raw_reward = np.std(self.raw_rewards[-self.window_size:])
                
                # Log metrics
                with open(f"{self.log_dir}/training_metrics.txt", "a") as f:
                    f.write(f"Episode {self.episode_count}: Mean Reward = {mean_raw_reward:.2f}, "
                           f"Std = {std_raw_reward:.2f}, Length = {episode_length}\n")
                
                # Check for convergence
                if mean_raw_reward > self.reward_threshold and std_raw_reward < self.std_threshold:
                    print(f"Converged at episode {self.episode_count}: "
                          f"Mean Reward = {mean_raw_reward:.2f}, Std = {std_raw_reward:.2f}")
                    return False  # Stop training
        return True

    def _on_training_end(self):
        # Save rewards data
        if self.raw_rewards:
            np.savetxt(f"{self.log_dir}/raw_rewards.csv", self.raw_rewards, delimiter=",")

def plot_learning_curve(log_dir, title, save_path):
    """Plot learning curve from training logs"""
    try:
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"No data found in {log_dir}/monitor.csv")
            return
        
        x, y = ts2xy(results, 'timesteps')
        
        plt.figure(figsize=(12, 8))
        
        # Plot raw rewards
        plt.subplot(2, 1, 1)
        plt.plot(x, y, alpha=0.6, color='blue', label="Episode Rewards")
        
        # Add moving average
        if len(y) > 50:
            window = min(50, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            x_avg = x[window-1:]
            plt.plot(x_avg, moving_avg, color='red', linewidth=2, label=f"Moving Average ({window})")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(f'{title} - Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot episode lengths
        plt.subplot(2, 1, 2)
        episode_lengths = results['l'].values
        plt.plot(range(len(episode_lengths)), episode_lengths, alpha=0.6, color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {save_path}")
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

def train_dqn_optimized():
    """Train DQN with optimized hyperparameters"""
    os.makedirs("models/dqn", exist_ok=True)
    log_dir = "logs/dqn"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting DQN Training")
    
    # Create environment
    env = EnhancedAgrikaTractorFleetEnv()  # Updated to Enhanced environment
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    
    # Optimized hyperparameters from the report
    learning_rate = 0.0005
    buffer_size = 100000
    batch_size = 64
    gamma = 0.99
    exploration_fraction = 0.5
    exploration_final_eps = 0.05
    target_update_interval = 1000
    train_freq = 4
    
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": 1000,
        "batch_size": batch_size,
        "tau": 0.001,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": 1,
        "target_update_interval": target_update_interval,
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": exploration_final_eps,
        "max_grad_norm": 10,
        "tensorboard_log": log_dir,
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "verbose": 1,
        "device": "auto"
    }
    
    print(f"Using hyperparameters: {model_params}")
    model = DQN(**model_params)
    
    # Setup evaluation
    eval_env = Monitor(EnhancedAgrikaTractorFleetEnv(), filename=f"{log_dir}/eval_monitor.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    training_callback = TrainingCallback(log_dir=log_dir)
    
    # Train the model
    total_timesteps = 150000
    print(f"Training DQN for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, training_callback],
            progress_bar=True,
            tb_log_name="dqn_optimized"
        )
        
        # Save final model
        model.save(f"{log_dir}/final_dqn_model")
        
        # Generate learning curve
        plot_learning_curve(log_dir, "DQN Learning Curve", 
                            f"{log_dir}/dqn_learning_curve.png")
        
        print(f"✅ DQN training completed!")
        
        # Evaluate final performance
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        env.close()
        eval_env.close()
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_dqn_optimized()
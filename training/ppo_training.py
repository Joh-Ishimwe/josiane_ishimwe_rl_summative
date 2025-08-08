import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import gymnasium

# Import the environment
from environment.agrika_env import EnhancedAgrikaTractorFleetEnv

class TrainingCallback(BaseCallback):
    """Custom callback to track training metrics and detect convergence"""
    def __init__(self, log_dir, window_size=100, reward_threshold=220, std_threshold=30):
        super().__init__(verbose=1)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self):
        # Handle vectorized environments
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])
        
        # Check if any environment finished an episode
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.episode_count += 1
                episode_reward = info.get('episode_reward', 0.0)
                episode_length = info.get('episode_length', self.locals['n_steps'])
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Check convergence
                if len(self.episode_rewards) >= self.window_size:
                    mean_reward = np.mean(self.episode_rewards[-self.window_size:])
                    std_reward = np.std(self.episode_rewards[-self.window_size:])
                    
                    # Log metrics
                    with open(os.path.join(self.log_dir, "training_metrics.txt"), "a") as f:
                        f.write(f"Episode {self.episode_count}: Mean Reward = {mean_reward:.2f}, "
                               f"Std = {std_reward:.2f}, Length = {episode_length}\n")
                    
                    # Check for convergence
                    if mean_reward >= self.reward_threshold and std_reward <= self.std_threshold:
                        print(f"Converged at episode {self.episode_count}: "
                              f"Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
                        return False  # Stop training
        
        return True

def train_ppo_optimized():
    """Train PPO with optimized hyperparameters"""
    # Create directories relative to script location
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models", "pg", "ppo")
    log_dir = os.path.join(base_dir, "logs", "ppo")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("🚀 Starting PPO Training with Optimized Hyperparameters")
    
    # Create vectorized environment with Monitor wrapper for each env
    def make_env():
        env = EnhancedAgrikaTractorFleetEnv()
        print(f"Created environment: {type(env)}")  # Debug: Confirm environment type
        if not isinstance(env, gymnasium.Env):
            raise ValueError(f"Environment is not a Gymnasium environment: {type(env)}")
        return Monitor(env, filename=os.path.join(log_dir, "monitor"))
    
    # Use 4 parallel environments
    n_envs = 4
    env = make_vec_env(make_env, n_envs=n_envs)
    
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
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=0.0
        ),
        tensorboard_log=None,
        verbose=1,
        device="cpu"
    )
    
    print(f"PPO Model created with {n_envs} parallel environments")
    print(f"Policy network: {model.policy}")
    
    # Setup evaluation
    eval_env = Monitor(EnhancedAgrikaTractorFleetEnv(), filename=os.path.join(log_dir, "eval_monitor.csv"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    # Setup training callback for convergence detection
    training_callback = TrainingCallback(log_dir=log_dir)
    
    # Train the model
    total_timesteps = 500000
    print(f"Training PPO for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, training_callback],
            progress_bar=False,  # Disable progress bar to avoid tqdm/rich dependency
            tb_log_name="ppo_optimized"
        )
        
        # Save final model
        model.save(os.path.join(model_dir, "final_ppo_model"))
        
        # Generate learning curve
        plot_learning_curve(log_dir, "PPO Learning Curve", os.path.join(model_dir, "ppo_learning_curve.png"))
        
        print("✅ PPO training completed!")
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Test generalization on different scenarios
        print("\n🧪 Testing generalization...")
        test_scenarios = [
            {'max_weather_changes': 2, 'breakdown_multiplier': 0.5},
            {'max_weather_changes': 8, 'breakdown_multiplier': 1.5}
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"Scenario {i+1}: {scenario}")
            test_env = EnhancedAgrikaTractorFleetEnv()
            test_rewards = []
            for _ in range(10):
                obs, _ = test_env.reset(options=scenario)
                episode_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                test_rewards.append(episode_reward)
            print(f"  Mean reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
            test_env.close()
        
        env.close()
        eval_env.close()
        return model
        
    except Exception as e:
        print(f"❌ PPO training failed: {str(e)}")
        env.close()
        eval_env.close()
        raise

def plot_learning_curve(log_dir, title, save_path):
    """Plot learning curve from training logs"""
    try:
        monitor_path = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(monitor_path):
            print(f"No data found in {monitor_path}")
            return
        
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"Empty data in {monitor_path}")
            return
        
        x, y = ts2xy(results, 'timesteps')
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(x, y, alpha=0.6, color='blue', label="Episode Rewards")
        
        # Add moving average
        if len(y) > 50:
            window = min(100, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            x_avg = x[window-1:]
            plt.plot(x_avg, moving_avg, color='red', linewidth=2, label=f"Moving Average ({window})")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(f'{title} - Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        plt.subplot(2, 2, 2)
        episode_lengths = results['l'].values
        plt.plot(range(len(episode_lengths)), episode_lengths, alpha=0.6, color='green')
        if len(episode_lengths) > 50:
            window = min(50, len(episode_lengths) // 10)
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_lengths)), moving_avg, color='red', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(y, bins=50, alpha=0.7, color='purple')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training progress (reward vs episodes)
        plt.subplot(2, 2, 4)
        episodes = range(len(y))
        plt.plot(episodes, y, alpha=0.6, color='orange')
        if len(y) > 50:
            window = min(50, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(y)), moving_avg, color='red', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

if __name__ == "__main__":
    model = train_ppo_optimized()
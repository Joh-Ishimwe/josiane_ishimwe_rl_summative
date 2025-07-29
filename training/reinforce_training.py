import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
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

def plot_learning_curve(log_dir="logs/reinforce", output_path="logs/reinforce/learning_curve.png"):
    """Plot learning curve from monitor logs"""
    try:
        results = load_results(log_dir)
        if len(results) == 0:
            print(f"No data found in {log_dir}/monitor.csv")
            return
        x, y = ts2xy(results, 'timesteps')
        raw_y = (y + 1) / 2 * (500 - (-600)) + (-600)  # Convert normalized to raw
        plt.figure(figsize=(10, 6))
        plt.plot(x, raw_y, label="Mean Episode Raw Reward", color="#1f77b4")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Raw Reward")
        plt.title("REINFORCE-like Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")

def train_reinforce_like(trial=None, total_timesteps=500000):
    """Train REINFORCE-like agent using A2C with optimized settings"""
    os.makedirs("models/pg/reinforce", exist_ok=True)
    log_dir = f"logs/reinforce/trial_{trial.number}" if trial else "logs/reinforce"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting REINFORCE-like Training (Trial {trial.number if trial else 'Default'})")
    print("Note: Using A2C with settings approximating REINFORCE")
    
    env = make_vec_env(AgrikaTractorFleetEnv, n_envs=1, monitor_dir=log_dir, env_kwargs={"normalize_rewards": True})
    eval_env = Monitor(AgrikaTractorFleetEnv(normalize_rewards=True), filename=f"{log_dir}/eval_monitor.csv")
    
    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4]) if trial else 0.0005,
        "n_steps": 60,  # Matches season_length
        "gamma": 0.99,
        "gae_lambda": 1.0,  # REINFORCE-like
        "ent_coef": trial.suggest_categorical("ent_coef", [0.03, 0.05, 0.1]) if trial else 0.03,
        "vf_coef": trial.suggest_categorical("vf_coef", [0.25, 0.5]) if trial else 0.5,
        "max_grad_norm": 0.5,
        "use_rms_prop": False,  # Use Adam
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[128, 64],  # Policy network
                vf=[128, 64]   # Larger value network
            )
        ),
        "tensorboard_log": log_dir,
        "verbose": 1,
        "device": "auto"
    }
    
    model = A2C(**model_params)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=20,  # More episodes for reliable std dev
        render=False
    )
    
    training_callback = TrainingCallback(log_dir=log_dir)
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, training_callback],
        progress_bar=True,
        tb_log_name=f"reinforce_like_trial_{trial.number}" if trial else "reinforce_like"
    )
    
    model.save(f"{log_dir}/final_reinforce_model")
    env.close()
    eval_env.close()
    
    print(f"✅ Training completed for trial {trial.number if trial else 'Default'}!")
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, return_episode_rewards=True)
    raw_rewards = [(r + 1) / 2 * (500 - (-600)) + (-600) for r in mean_reward]
    return np.mean(raw_rewards) - np.std(raw_rewards)  # Objective for optuna

def tune_reinforce_like():
    """Run hyperparameter tuning with optuna"""
    study = optuna.create_study(direction="maximize")
    study.optimize(train_reinforce_like, n_trials=3)  # Reduced to 3 trials
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    return study

if __name__ == "__main__":
    study = tune_reinforce_like()
from wandb.integration.sb3 import WandbCallback
import mobile_env 
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb



#model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu')
#model.learn(total_timesteps=10000000, callback=WandbCallback())


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder



config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500000,
    "env_name": "mobile-small-central-v0",
}
run = wandb.init(
    project="MOBILE-ENV",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
"""env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)"""
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()

model.save('ppo.pth')


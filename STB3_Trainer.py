import gymnasium as gym
import StockEnv.StockEnv
import MTXEnv
from StockEnv.envs import StockEnvV2
import numpy as np
import torch
import random

import time
from argparse import ArgumentParser
#from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import obs_as_tensor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from masking import action_mask_fn
import os
import torch.backends.cudnn as cudnn
import yaml
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
cudnn.benchmark = True
SEED = 42


class PhaseTimer(BaseCallback):
    def __init__(self): 
        super().__init__()
        self.collect_t = None
        self.train_t = None
        self.rollout_count = 0
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_start(self):
        # 如果有上一輪的訓練時間，先打印
        if self.train_t is not None:
            print(f"[timer] train_sec={time.time()-self.train_t:.2f}")
        
        # 開始新的數據收集
        self.collect_t = time.time()
        
    def _on_rollout_end(self):
        if self.collect_t is not None:
            print(f"[timer] collect_sec={time.time()-self.collect_t:.2f}")
            self.rollout_count += 1
            # 訓練即將開始
            self.train_t = time.time()
            print(f"[timer] training_start - rollout #{self.rollout_count}")
            
    def _on_training_start(self): 
        # 這個方法在 MaskablePPO 中可能不會被調用
        pass
        
    def _on_training_end(self):
        # 這個方法在 MaskablePPO 中可能不會被調用
        pass

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        print(progress)
        return progress * initial_value

    return func

def lr_cosine_with_min(init=2e-4, min_lr=5e-5):

    import math
    def f(progress_remaining: float) -> float:
        p = max(0.0, min(1.0, progress_remaining))           # 1→0
        cos_w = (1 + math.cos(math.pi * (1 - p))) / 2        # 1→0
        return float(min_lr + (init - min_lr) * cos_w)
    return f


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    

def mask_fn(env):
    return env.valid_action_mask()
                
def make_env(env_id, save_dir, sim_start, sim_end, train_start, train_end, init_balance, mask=False):
        
    # environmen
    env = gym.make(env_id, save_dir=save_dir, sim_start=sim_start, sim_end=sim_end, train_start=train_start, train_end=train_end, init_balance=init_balance)
    if mask:
        env = ActionMasker(env, mask_fn)

    return env 

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--save_name', type=str, help='The name of file to save.')
    parser.add_argument('--env', type=str, help='The name of the env.')
    parser.add_argument('--config', type=str, help="config file for times")
    parser.add_argument('--balance', type=int, help='initial balance of agent.')
    parser.add_argument('-m', '--mask', action='store_true', help='using maskable env.')
    parser.add_argument('--retrain', action='store_true', help='retrain the model.')
    parser.add_argument('--trained_model', type=str, help='name of the retrain model.')
    parser.add_argument('--total_step', type=int, help="total time step to train")
    parser.add_argument('--data_dir', type=str, default="processed_data", help="data directory")
    args = parser.parse_args()
    config = load_config(args.config)
    hypers = config["hyperparameter"]
    dates = config['date']
    feat_columns = config.get('feat', None)  # Get feat columns from config
    print(f'Training config : ')
    print(f'Learing rate : {hypers["lr"]}')
    print(f'Batch size : {hypers["batch_size"]}')
    print(f'Total timesteps : {args.total_step}')
    print(f'N_epoch : {hypers["n_epoch"]}')
    print(f'Gamma : {hypers["gamma"]}')
    print(f'Steps : {hypers["n_steps"]}')
    print(f'Initial balance : {args.balance}')
    print(f'Save name : {args.save_name}')
    print(f'Env : {args.env}')
    print(f'Feature columns : {feat_columns}')
    print(f'Data directory : {args.data_dir}')
    print(f'='*165)

    # train the model    
    seed_everything(SEED)
    env_kwargs = dict(
        save_dir=f'trained_csv/{args.save_name}',
        sim_start=dates["sim_start"],
        sim_end=dates["sim_end"],
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        init_balance=args.balance,
        feat_columns=feat_columns,
        data_dir=args.data_dir
    )
    env = make_vec_env(
        args.env,                       # <--- 直接給註冊好的環境 ID
        n_envs=hypers['n_envs'],
        vec_env_cls=SubprocVecEnv,      # Windows 必用子程序
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker if args.mask else None,
        wrapper_kwargs={"action_mask_fn": action_mask_fn} if args.mask else None,
    )
    start_time = time.time()
    
    if args.retrain:
        model = MaskablePPO.load(f'trained_model/{args.trained_model}.zip', env=env)
    else:
        model = MaskablePPO('MlpPolicy', 
                    env, 
                    ent_coef=0.005, 
                    batch_size=hypers['batch_size'], 
                    learning_rate=linear_schedule(hypers['lr']), 
                    n_epochs=hypers['n_epoch'], 
                    gamma=hypers['gamma'], 
                    n_steps=hypers['n_steps'], 
                    verbose=1,
                )     
    print("device:", model.device)

    model.learn(total_timesteps=args.total_step, callback=PhaseTimer())
    model.save(f'trained_model/{args.save_name}')
    end_time = time.time()
    print(f'training time : {end_time-start_time}s')
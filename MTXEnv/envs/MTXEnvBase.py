import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import os
import math
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from abc import ABC,abstractmethod
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
from bisect import bisect_right
def data_to_numpy(df):
    time_arr = df["date"].values.astype("datetime64[ns]")
    feat_df = df.drop(columns=["date"] + (None or []))
    feat_arr = feat_df.to_numpy(dtype=np.float32)
    col_index = {col: i for i, col in enumerate(feat_df.columns)}
    return time_arr, feat_arr, col_index
class MonthIterator:
    def __init__(self, start: str, end: str, infinite: bool = True):
        """
        start, end: "YYYY-MM" 格式
        infinite: True = 無限循環, False = 單輪
        """
        self.start_dt = datetime.strptime(start, "%Y%m")
        self.end_dt   = datetime.strptime(end, "%Y%m")
        self.infinite = infinite
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur > self.end_dt:
            if self.infinite:
                self.cur = self.start_dt
            else:
                raise StopIteration
        val = self.cur.strftime("%Y%m")
        self.cur += relativedelta(months=1)
        return val

    def reset(self):
        """重設回起始月"""
        self.cur = self.start_dt

# Base class for the stock market environment
class MTXEnvBase(gym.Env, ABC):
    """
    Base class for the stock market trading environment.

    This class sets up a trading environment using OpenAI Gym, allowing reinforcement learning agents
    to interact with a simulated stock market. It includes functionality for trade execution, state 
    management, normalization, and performance evaluation.

    Attributes:
        CONTRACT_PRICE (float): The fixed price per contract.
        REWARD_SCALING (float): Scaling factor for the reward.
        TRANSACTION_FEE_PERCENT (float): Transaction fee as a percentage of the trade.
        POINT_VALUE (float): The point value used for reward calculations.
        RISK_FREE (float): Risk-free return rate used in performance evaluation.
        MAX_SHARES (int): Maximum number of shares that can be owned.

    Methods:
        setup_spaces(): Sets up the action and observation spaces.
        find_min_max(): Computes min and max values for data normalization.
        load_entire_data(): Loads market data for normalization purposes.
        normalize(value, min_val, max_val): Normalizes a value between 0 and 1 based on provided min and max.
        reset_environment(): Resets the month and year for simulation.
        reset(seed=None, options=None): Resets the environment to its initial state and returns the initial observation.
        get_observation(): Returns the current normalized observation space.
        step(action): Takes a step in the environment based on the provided action (to be implemented in derived class).
        process_trade(action): Executes a trade based on the action (to be implemented in derived class).
        calculate_asset(): Calculates the total asset value (to be implemented in derived class).
        finalize_episode(asset): Finalizes the episode when done, computes the reward, and updates balance and trades.
        record_info(asset, action, reward): Records the current state information in the DataFrame.
        save_results(): Saves the simulation results to the specified file system location.
        get_info(): Provides information about the current state of the environment.
        get_performance(): Evaluates the performance of the agent in the environment.
        render(mode='human'): Renders the environment (optional, can be expanded).
    """
    
    

    def __init__(self, save_dir, train_start, train_end, sim_start, sim_end, init_balance, feat_columns=None, mode="Train", data_dir="processed_data") -> None:
        """
        Initializes the stock market environment with the given parameters.

        Args:
            save_dir (str): Directory path for saving results.
            train_start (str): Start time for training data.
            train_end (str): End time for training data.
            sim_start (str): Initial time for simulation.
            sim_end (str): Final time for simulation.
            init_balance (float): Initial balance for the agent (default is 1,000,000).
            feat_columns (list): List of feature columns to use for observation (default: ["close", "macd", "MFI", "cci", "adx"]).
            mode (str): Mode of operation ("Train" or "Test").
            data_dir (str): Directory path for training data (default: "processed_data").
        """
        
        super().__init__()
        # Constant values used in the environment
        self.CONTRACT_PRICE = 33250
        self.INITIAL_MARGIN = 33250
        self.MAINTENANCE_MARGIN = 25500
        self.REWARD_SCALING = 1e-4
        self.TRANSACTION_FEE_PERCENT = 0.00002
        self.POINT_VALUE = 50
        self.RISK_FREE = 1.7
        self.MAX_SHARES = 100
        self.WINDOW_SIZE = 76       
        # Initialize environment parameters
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.train_start = train_start
        self.train_end = train_end
        self.scalar = 15
        
        # Set feature columns for observation
        if feat_columns is None:
            self.feat_columns = ["close", "macd", "mfi", "cci", "adx"]
        else:
            self.feat_columns = feat_columns
        
        self.setup_spaces() # Setup action and observation spaces
        self.mode = mode
        
        # Initialize state variables
        self.position = 'None' # Position can be 'long', 'short', or 'None'
        self.win = 0 # Number of successful trades
        self.totalRound = 0 # Total number of trades
        
        self.sim_cur = None
        self.index = None # Current index in the DataFrame
        self.df = None # DataFrame containing market data
        self.df_all = {}
        self.time_arr_all = {}
        self.feat_arr_all = {}
        self.col_index_all = {}
        self.balance = init_balance # Current balance of the agent
        self.init_balance = init_balance # Initial balance
        self.shares = 0 # Number of shares owned
        self.total_reward = 0 # Total reward accumulated
        self.last_bid = [] # List of last bid prices
        self.mergin = 0
        
        # Initialize min and max values for normalization
        self.close_max = -np.inf
        self.volume_max = -np.inf
        self.rsi_max = -np.inf
        self.cci_max = -np.inf
        self.macd_max = -np.inf
        self.adx_max = -np.inf
        self.mfi_max = -np.inf
        self.close_min = np.inf
        self.volume_min = np.inf
        self.rsi_min = np.inf
        self.cci_min = np.inf
        self.macd_min = np.inf
        self.adx_min = np.inf
        self.mfi_min = np.inf
        self.month_iter = MonthIterator(start=sim_start, end=sim_end, infinite=True)
        self.margin_time = None
        self.margin_value = None
        self.getmargin()
        self.find_min_max() # Find min and max values for normalization
        #self.print_param()
        
    def getmargin(self):
        df = pd.read_csv("Margin_Schedule.csv", parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        self.margin_time = df['time'].values.astype("datetime64[ns]")
        self.margin_value = df[["contract_price", "initial_margin", "maintenance_margin"]].values
        keys = self.margin_time.view('int64')
        self.M = {k: v for k, v in zip(keys, self.margin_value)}

    def print_param(self):
        print(f'CONTRACT PRICE : {self.CONTRACT_PRICE}')
        print(f'INITIAL MARGIN : {self.INITIAL_MARGIN}')
        print(f'MAINTENANCE MARGIN: {self.MAINTENANCE_MARGIN}')
        print(f'REWARD SCALING : {self.REWARD_SCALING}')
        print(f'TRANSACTION FEE PERCENT : {self.TRANSACTION_FEE_PERCENT}')
        print(f'POINT VALUE : {self.POINT_VALUE}')
        print(f'RISK FREE : {self.RISK_FREE}')
        print(f'MAX SHARES : {self.MAX_SHARES}')
        print(f'WINDOW SIZE : {self.WINDOW_SIZE}')
        print(f'close max : {self.close_max}')
        print(f'close min : {self.close_min}')
        print(f'volume max : {self.volume_max}')
        print(f'volume min : {self.volume_min}')
        print(f'rsi max : {self.rsi_max}')
        print(f'rsi min : {self.rsi_min}')
        print(f'cci max : {self.cci_max}')
        print(f'cci min : {self.cci_min}')
        print(f'macd max : {self.macd_max}')
        print(f'macd min : {self.macd_min}')
        print(f'adx max : {self.adx_max}')
        print(f'adx min : {self.adx_min}')
        print(f'train start : {self.train_start}')
        print(f'train end : {self.train_end}')
    
    def setup_spaces(self):
        # Define action space [-100, 100] discrete and observation space (feature columns * window size, balance, shares)
        self.action_space = spaces.Discrete(201)  # [-100, 100] -> 201 actions
        # Calculate observation space size based on number of feature columns
        obs_size = len(self.feat_columns) * self.WINDOW_SIZE + 2  # +2 for balance and shares
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,))        

        
    def find_min_max(self):
        # Load entire dataset to find min and max for normalization
        whole_df = self.load_entire_data()
        
        # Dynamically calculate min/max for all feature columns
        for col in self.feat_columns:
            if col in whole_df.columns:
                setattr(self, f"{col.lower()}_max", np.max(whole_df[col]))
                setattr(self, f"{col.lower()}_min", np.min(whole_df[col]))
            else:
                # Set default values if column doesn't exist
                setattr(self, f"{col.lower()}_max", 1.0)
                setattr(self, f"{col.lower()}_min", 0.0)
                print(f"Warning: Column '{col}' not found in data, using default min/max values")
        
        # Keep volume for backward compatibility (if needed)
        if 'volume' in whole_df.columns:
            self.volume_max = np.max(whole_df['volume'])
            self.volume_min = np.min(whole_df['volume'])
        
        # Pre-normalize feat_arr_all only in Train mode
        if hasattr(self, 'mode') and self.mode == "Train":
            self._pre_normalize_feat_arr_all()

    def _pre_normalize_feat_arr_all(self):
        """
        Pre-normalize all feat_arr_all data for faster get_observation in Train mode.
        This creates normalized versions with only feat_columns data in observation order.
        """
        if not hasattr(self, 'feat_arr_all') or self.feat_arr_all is None:
            return
        
        # Initialize feat_arr_all_normalized dictionary
        self.feat_arr_all_normalized = {}
        
        # Normalize each dataset in feat_arr_all
        for key, feat_arr in self.feat_arr_all.items():
            if feat_arr is not None:
                # Get the corresponding col_index
                col_index = self.col_index_all.get(key, {})
                
                # Create normalized array with only feat_columns data in observation order
                # Shape: (time_steps, len(feat_columns))
                feat_arr_normalized = np.zeros((feat_arr.shape[0], len(self.feat_columns)), dtype=np.float32)
                
                # Normalize each feature column in the correct order
                for feat_idx, col in enumerate(self.feat_columns):
                    if col in col_index:
                        col_idx = col_index[col]
                        min_val = getattr(self, f"{col.lower()}_min", 0)
                        max_val = getattr(self, f"{col.lower()}_max", 1)
                        
                        # Avoid division by zero
                        if max_val != min_val:
                            feat_arr_normalized[:, feat_idx] = (feat_arr[:, col_idx] - min_val) / (max_val - min_val)
                        else:
                            feat_arr_normalized[:, feat_idx] = 0.0
                    # else remains 0.0 (already initialized)
                
                # Store the normalized data
                self.feat_arr_all_normalized[key] = feat_arr_normalized

    def load_entire_data(self):
        """
        Loads the entire training dataset for normalization.

        Returns:
            pd.DataFrame: Concatenated DataFrame of the entire dataset for the specified date range.
        """
        whole_df = pd.DataFrame()
        
        mi = MonthIterator(self.train_start, self.train_end, infinite=False)
        for m in mi:
            file_path = f'{self.data_dir}/mtx-{m}-{self.scalar}min.csv'
            key = m
            temp_df = pd.read_csv(file_path)
            time_arr, feat_arr, col_index = data_to_numpy(temp_df)
            self.df_all[key] = temp_df
            self.time_arr_all[key] = time_arr
            self.feat_arr_all[key] = feat_arr
            self.col_index_all[key] = col_index
            whole_df = pd.concat([whole_df, temp_df], axis=0, ignore_index=True)
        return whole_df 
    
    def normalize(self, value, min_val, max_val):
        """
        Normalizes a value between 0 and 1 based on provided min and max values.

        Args:
            value (float): The value to be normalized.
            min_val (float): The minimum value for normalization.
            max_val (float): The maximum value for normalization.

        Returns:
            float: Normalized value between 0 and 1.
        """        

        # Normalize a value between 0 and 1 based on min and max
        return (value - min_val) / (max_val - min_val)
    
    def reset_environment(self):
        self.sim_cur = next(self.month_iter)
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        pass
    
    def update_contract_price(self):
        ts = self.time_arr[self.index]
        price = self.M.get(ts.view('int64'))
        if price is not None:
            self.CONTRACT_PRICE = price[0]
            self.INITIAL_MARGIN = price[1]
            self.MAINTENANCE_MARGIN = price[2]

    def reset_prices(self):
        ts = self.time_arr[self.index]
        i = bisect_right(self.margin_time, ts) - 1
        if i < 0:
            return None
        self.CONTRACT_PRICE = self.margin_value[i][0]
        self.INITIAL_MARGIN = self.margin_value[i][1]
        self.MAINTENANCE_MARGIN = self.margin_value[i][2]

    def is_tradable(self):
        minutes = self.time_arr[self.index].astype("datetime64[m]").astype(int) % (24 * 60)
        if(minutes == 13*60+45 or minutes == 5*60):
            return False
        return True 
    
    def get_observation(self):
        """
        Returns the current normalized observation space.
        Uses pre-normalized data for faster performance in Train mode.

        Returns:
            np.ndarray: Array containing normalized state variables.
        """
        observation = [
            self.balance / (self.init_balance * 3),
            self.shares / 100,
        ]
        
        # Use pre-normalized data if available (Train mode), otherwise use original method
        if (hasattr(self, 'mode') and self.mode == "Train" and 
            hasattr(self, 'feat_arr_all_normalized') and self.feat_arr_all_normalized is not None and
            hasattr(self, 'curr_key') and self.curr_key in self.feat_arr_all_normalized):
            
            # Fast path: use pre-normalized data with direct slicing
            feat_arr_normalized = self.feat_arr_all_normalized[self.curr_key]
            
            # Create indices for all time steps (from newest to oldest)
            indices = np.arange(self.index, self.index - self.WINDOW_SIZE, -1)
            valid_mask = indices >= 0
            
            # Pre-allocate observation data array
            obs_data = np.zeros((self.WINDOW_SIZE, len(self.feat_columns)), dtype=np.float32)
            
            # Extract valid data using slicing
            valid_indices = indices[valid_mask]
            if len(valid_indices) > 0:
                # Use advanced indexing to get all valid data at once
                # This maintains the correct time order
                obs_data[:len(valid_indices), :] = feat_arr_normalized[valid_indices, :]
            
            # Flatten in the correct order (row-wise, time-wise)
            observation.extend(obs_data.flatten())
        else:
            # Fallback: use original normalization method
            for i in range(self.WINDOW_SIZE):
                index_offset = self.index - i
                if index_offset >= 0:  
                    for col in self.feat_columns:
                        # Get min/max values for this column
                        min_val = getattr(self, f"{col.lower()}_min", 0)
                        max_val = getattr(self, f"{col.lower()}_max", 1)
                        
                        # Normalize the value
                        normalized_val = self.normalize(
                            self.feat_arr[index_offset, self.col_index[col]], 
                            min_val, 
                            max_val
                        )
                        observation.append(normalized_val)
                else:
                    # Handle edge case when index_offset < 0
                    for col in self.feat_columns:
                        observation.append(0.0)  # Fill with zeros for invalid indices
        
        return np.array(observation, dtype=np.float32)
    @abstractmethod
    def step(self, action):
        """Take a step in the environment."""
        pass
        # return observation, reward, done, False, self.get_info()

    @abstractmethod
    def process_trade(self, action):
        """Executes a trade based on the action."""
        pass

    @abstractmethod
    def calculate_asset(self):
        """Calculates the total asset value."""
        pass


    def finalize_episode(self, asset):
        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        
        self.balance += self.CONTRACT_PRICE * self.shares
        self.balance -= (self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT) * self.shares
        for i in range(len(self.last_bid)):
            self.balance += (self.df.iloc[self.index]['close'] - self.last_bid.pop(0)) * self.POINT_VALUE
        
        reward = (self.balance - asset) * self.REWARD_SCALING
        
        self.index += 1 
        self.record_info(self.balance, -self.shares, reward)
        self.index -= 1 
        
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        print(f'win rate : {self.win / self.totalRound}')
        # Saving the results
        self.save_results()

        return reward
    
    def record_info(self, asset, action, reward):
        """
        Records the current state information in the DataFrame.

        Args:
            asset (float): The total asset value.
            action (float): Action taken by the agent.
            reward (float): Reward received for the action.
        """
        if self.mode != "Train": # Record the current state information in the DataFrame
            self.feat_arr[self.index-1, self.col_index["balance"]] = self.balance
            self.feat_arr[self.index-1, self.col_index["asset"]] = asset
            self.feat_arr[self.index-1, self.col_index["shares"]] = self.shares
            self.feat_arr[self.index-1, self.col_index["action"]] = action
            self.feat_arr[self.index-1, self.col_index["reward"]] = reward
        
    def save_results(self):
        """Saves the simulation results to the file system."""
        if self.mode != "Train":
            #Saves the simulation results to the file system.
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'mtx-{self.sim_cur}-{self.scalar}min.csv')
            self.df.to_csv(save_path, index=False)

    def get_info(self):
        """Provides information about the current state."""
        return {
            'balance': self.balance,
            'shares': self.shares,
            'date': self.time_arr[self.index],
        }
        
    def get_performance(self):
        """
        Evaluates the performance of the agent in the environment.

        Returns:
            float: The return on investment (RoR).
            float: The Sharpe ratio.
            float: The maximum drawdown.
        """
        RoR = (self.df.iloc[-1]['asset'] - self.df.iloc[0]['asset']) / self.df.iloc[0]['asset']
        mean_return = self.df['return'].mean() * 19 * 21 * 4
        std_return = self.df['return'].std() * ((19 * 21 * 4) ** 0.5)
        Sharp = (mean_return - self.RISK_FREE) / std_return
        
        cumulative_return = (self.df['cumulative return'].values / 100) + 1
        peak = np.maximum.accumulate(cumulative_return)
        peak = np.where(peak == 0, np.nan, peak)
        drawdown = (cumulative_return - peak) / peak
        drawdown = np.nan_to_num(drawdown, nan=0)

        Max_drawdown = drawdown.min() * 100
        
        return RoR, Sharp, Max_drawdown
    
    def render(self, mode='human'):
        """Renders the environment (optional, can be expanded)."""
        pass
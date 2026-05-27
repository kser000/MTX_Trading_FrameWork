from .MTXEnvSyn import *
from datetime import datetime
from .Indicator import *
import pickle
class MTXEnvRealTimeTrader(MTXEnvSyn):
    def __init__(self, save_dir, train_start, train_end, sim_start, sim_end, init_balance, margins, feature_columns=None, mode="Test", data_dir="processed_data"):
        super().__init__(save_dir, train_start, train_end, sim_start, sim_end, init_balance, feature_columns, mode, data_dir)
        self.curr_trader = None
        self.indicator_state = None
        self.CONTRACT_PRICE = margins[0]
        self.INITIAL_MARGIN = margins[1]
        self.MAINTENANCE_MARGIN = margins[2]
        # self.trader_list = []
        
    def reset_environment(self):
        """Resets month and year for simulation."""
        self.sim_cur = self.sim_start

    def record_info(self, asset, action, reward):
        self.df.loc[self.index, 'balance'] = self.balance
        self.df.loc[self.index, 'asset'] = asset
        self.df.loc[self.index, 'shares'] = self.shares
        self.df.loc[self.index, 'action'] = action
        self.df.loc[self.index, 'reward'] = reward

    def get_observation(self):

        observation = [
            self.balance / (self.init_balance * 3),
            self.shares / 100,
        ]
        

        for i in range(self.WINDOW_SIZE):
            index_offset = self.index - i
            if index_offset >= 0:  
                observation.extend([
                    self.normalize(self.df.iloc[index_offset]['close'], self.close_min, self.close_max),
                    self.normalize(self.df.iloc[index_offset]['macd'], self.macd_min, self.macd_max),
                    self.normalize(self.df.iloc[index_offset]['mfi'], self.mfi_min, self.mfi_max),
                    #self.normalize(self.df.iloc[index_offset]['rsi'], self.rsi_min, self.rsi_max),
                    self.normalize(self.df.iloc[index_offset]['cci'], self.cci_min, self.cci_max),
                    self.normalize(self.df.iloc[index_offset]['adx'], self.adx_min, self.adx_max),
                ])
            else:
                print(f'There is a problem with the index.')

        
        return np.array(observation, dtype=np.float32)
    
    def get_info(self):

        return {
            'date': self.df.iloc[self.index]['date'],
            'balance': self.balance,
            'asset': self.df.iloc[self.index]['asset'],
            'shares': self.shares,
            'close': self.df.iloc[self.index]['close'],
            'action': self.df.iloc[self.index]['action'],
        }

    def set_trader(self, trader):
        self.curr_trader = trader

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generator (default is None).
            options (dict, optional): Additional options for resetting the environment (default is None).

        Returns:
            np.ndarray: An array containing the initial observation of the environment.
        """
        self.reset_environment()
        self.curr_trader = options
        file_path = f'{self.save_dir}/mtx-{self.sim_cur}-{self.scalar}min.csv'
        print(f'using {file_path}.')
        print(f'curr time : {self.sim_cur}, contract price : {self.CONTRACT_PRICE}')
        self.df = pd.read_csv(file_path)
        self.indicator_state = init_state_from_df(self.df)
        self.position = 'None'
        self.index = len(self.df) - 1
        self.balance = self.df.iloc[self.index]['balance']
        self.shares = int(self.df.iloc[self.index]['shares'])
        self.total_reward = self.df['reward'].sum()
        with open(f"{self.save_dir}/{self.sim_cur}.pkl", "rb") as f:
            self.last_bid = pickle.load(f)

        # self.trader_list = []
        self.win = 0
        self.totalRound = 0

        self.long_win = 0
        self.short_win = 0
        self.long_trade_round = 0
        self.short_trade_round = 0
        
        observation = self.get_observation()
        return observation, self.get_info()     

    def update_state(self, kbar):
        self.df = append_bar_with_indicators(self.df, kbar, self.indicator_state)
        self.index += 1
        return self.get_observation()

    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0

        # record current position
        self.df.loc[self.index, 'trader'] = self.curr_trader

        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]

        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()       
        
        if begin_asset <= 0:
            done = True
            reward += self.finalize_episode()
            self.total_reward += reward
            print(f'total reward : {self.total_reward}')
            return None, reward, done, False, None
        # process trade
        reward = self.process_trade(action)
        self.total_reward += reward
        
        #get next obs
        observation = self.get_observation()
        #record some useful info
        self.record_info(begin_asset, action, reward)
            
        # update Contract Price 
        
        return observation, reward, done, False, self.get_info()
        
    def process_trade(self, action):
        """Executes a trade based on the action."""
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT + 25
        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy 
            # Long position   
            if self.shares >= 0:
                
                self.balance -= (self.CONTRACT_PRICE + transaction_fee) * action
                # Record the bid price
                for _ in range(action):
                    # Calculate win rate
                    self.totalRound += 1
                    self.long_trade_round += 1
                    # self.trader_list.append(1)
                    self.last_bid.append(self.df.iloc[self.index]['close'])
                                                   
                 
            # Short position
            elif self.shares < 0:
                self.balance += self.CONTRACT_PRICE * action
                self.balance -= transaction_fee * action           
                for _ in range(action):
                    bid_price = self.last_bid.pop(0)
                    self.balance += (bid_price - self.df.iloc[self.index]['close']) * self.POINT_VALUE
                    reward += bid_price - self.df.iloc[self.index]['close'] 
                    if bid_price > self.df.iloc[self.index]['close']:
                        self.win += 1
                        self.short_win += 1
            self.shares += action
        
        elif action < 0: # sell
            # Long position
            if self.shares > 0:
                self.balance += self.CONTRACT_PRICE * abs(action)
                self.balance -= transaction_fee * abs(action)           
                for _ in range(abs(action)):
                    bid_price = self.last_bid.pop(0)
                    self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
                    reward += self.df.iloc[self.index]['close'] - bid_price
                    if self.df.iloc[self.index]['close'] > bid_price:
                        self.win += 1
                        self.long_win += 1

            # Short postiion
            elif self.shares <= 0:
                self.balance -= (self.CONTRACT_PRICE + transaction_fee) * abs(action)
            
                # Record the bid price
                for _ in range(abs(action)):
                    # Calculate win rate
                    self.totalRound += 1
                    self.short_trade_round += 1
                    self.last_bid.append(self.df.iloc[self.index]['close']) 
                                   
            self.shares -= abs(action)
        else: # hold
            pass
        
        return reward
        
    def calculate_asset(self):
        """Calculates the total asset value."""
        asset = self.balance + self.CONTRACT_PRICE * abs(self.shares)
        for i in range(int(abs(self.shares))):
            if self.shares >= 0:
                asset +=  self.POINT_VALUE * (self.df.iloc[self.index]['close'] - self.last_bid[i])
            else:
                asset +=  self.POINT_VALUE * (self.last_bid[i] - self.df.iloc[self.index]['close'])
        return asset

    def is_tradable(self):
        cur_time = self.df.iloc[self.index]['date']
        if(cur_time.split()[1] == "13:45:00" or cur_time.split()[1] == "05:00:00"):
            return False
        return True
    
    def valid_action_mask_long(self):
        
        actions_mask = np.ones(201, dtype=int)
        if not self.is_tradable():
            actions_mask[0:100] = 0
            actions_mask[101:201] = 0
            return actions_mask.astype(bool)
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT + 25
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))
        available_amount = min(5-abs(self.shares), available_amount)
        '''curr_asset = self.calculate_asset()
        
        close_num = 0 # The number of shares to close
        # check wheather the margin requirements are met
        if curr_asset < self.maintenance_margin:
            available_amount = 0
            init_margin = self.initial_margin
            while(curr_asset < init_margin):
                close_num += 1
                init_margin -= self.INITIAL_MARGIN  '''      
    
        # long position
        if self.shares >= 0:                   
            # mask invalid action
            for i in range(101+available_amount, 201):
                actions_mask[i] = 0
            for i in range(0, 100-self.shares):
                actions_mask[i] = 0     
            #for i in range(101-close_num, 101):
                #actions_mask[i] = 0
        
        # short position
        elif self.shares < 0:      
            # mask invalid action
            for i in range(101+abs(self.shares), 201):
                actions_mask[i] = 0
            for i in range(0, min(100+abs(self.shares), 201)):
                actions_mask[i] = 0  
        return actions_mask.astype(bool)  

    def valid_action_mask_short(self):
        actions_mask = np.ones(201, dtype=int)
        if not self.is_tradable():
            actions_mask[0:100] = 0
            actions_mask[101:201] = 0
            return actions_mask.astype(bool)

        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT + 25
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))
        available_amount = min(5-abs(self.shares), available_amount)
        '''curr_asset = self.calculate_asset()
        
        close_num = 0 # The number of shares to close
        # check wheather the margin requirements are met
        if curr_asset < self.maintenance_margin:
            available_amount = 0
            init_margin = self.initial_margin
            while(curr_asset < init_margin):
                close_num += 1
                init_margin -= self.INITIAL_MARGIN '''  
        # long position
        if self.shares > 0:
            # mask invalid action
            for i in range(101-self.shares, 201):
                actions_mask[i] = 0
            for i in range(0, 100-self.shares):
                actions_mask[i] = 0
        # short position
        elif self.shares <= 0:        
            # masl invalid action
            for i in range(101+abs(self.shares), 201):
                actions_mask[i] = 0
            for i in range (0, 100-available_amount):
                actions_mask[i] = 0
            #for i in range(100, 100+close_num):
                #actions_mask[i] = 0
        return actions_mask.astype(bool)          

    def finalize_episode(self):

        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        reward = 0
        self.balance += self.CONTRACT_PRICE * abs(self.shares)
        self.balance -= (self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT + 25) * abs(self.shares)
        # Long position
        if self.shares >= 0:
            for i in range(len(self.last_bid)):
                bid_price = self.last_bid.pop(0)
                self.balance += (self.df.iloc[self.index]['close'] - bid_price) * self.POINT_VALUE
                reward += self.df.iloc[self.index]['close'] - bid_price
                if self.df.iloc[self.index]['close'] > bid_price:
                    self.win += 1
                    self.long_win += 1
        # Short position
        elif self.shares < 0:
            for i in range(len(self.last_bid)):
                bid_price = self.last_bid.pop(0)
                self.balance += (bid_price - self.df.iloc[self.index]['close']) * self.POINT_VALUE
                reward += bid_price - self.df.iloc[self.index]['close']
                if bid_price > self.df.iloc[self.index]['close']:
                    self.win += 1  
                    self.short_win += 1
        self.record_info(self.balance, -self.shares, reward) 
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        
        # Saving the results
        self.save_results()
        
            
        return reward
    
    def save_results(self):
        self.df['contract_month'] = self.sim_cur
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f'mtx-{self.sim_cur}-{self.scalar}min.csv')
        self.df.to_csv(save_path, index=False)
        with open(f"{self.save_dir}/{self.sim_cur}.pkl", "wb") as f:
            pickle.dump(self.last_bid, f)
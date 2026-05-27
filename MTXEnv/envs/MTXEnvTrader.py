from .MTXEnvSyn import *

class MTXEnvTrader(MTXEnvSyn):
    def __init__(self, save_dir, train_start, train_end, sim_start, sim_end, init_balance, feature_columns=None, mode="Test", data_dir="processed_data"):
        super().__init__(save_dir, train_start, train_end, sim_start, sim_end, init_balance, feature_columns, mode, data_dir)
        self.initial_margin = 0
        self.maintenance_margin = 0
        self.clearing_margin = 0
        
        self.long_win = 0
        self.short_win = 0
        self.long_trade_round = 0
        self.short_trade_round = 0
        self.short_open_long_close = 0
        self.long_open_short_close = 0
        self.curr_trader = None
        # self.trader_list = []
        
    def reset_environment(self):
        """Resets month and year for simulation."""
        self.sim_cur = next(self.month_iter)
    
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
        file_path = f'{self.data_dir}/mtx-{self.sim_cur}-{self.scalar}min.csv'
        

        self.df = pd.read_csv(file_path)
        self.df['balance'] = self.init_balance
        self.df['asset'] = self.init_balance
        self.df['shares'] = 0.0
        self.df['action'] = 0.0
        self.df['reward'] = 0.0
        self.df['trader'] = 0.0
        self.df_to_numpy()
        self.position = 'None'
        self.index = self.WINDOW_SIZE-1
        self.balance = self.init_balance
        self.shares = 0
        self.total_reward = 0
        self.last_bid = []
        
        self.win = 0
        self.totalRound = 0

        self.long_win = 0
        self.short_win = 0
        self.long_trade_round = 0
        self.short_trade_round = 0
        self.short_open_long_close = 0
        self.long_open_short_close = 0
        # self.trader_list = []
        self.curr_trader = None
        self.reset_prices()
        print(f'using {file_path}.')
        print(f'curr time: {self.sim_cur}, contract price : {self.CONTRACT_PRICE}')
        observation = self.get_observation()

        return observation, self.get_info(self.init_balance)

    def get_info(self, asset=None):
        info = {
            'balance': self.balance,
            'asset': asset,
            'shares': self.shares,
            'date': self.df.iloc[self.index]['date'],
        }
        return info

    def df_to_numpy(self):
        self.time_arr = self.df["date"].values.astype("datetime64[ns]")
        feat_df = self.df.drop(columns=["date"] + (None or []))
        self.feat_arr = feat_df.to_numpy(dtype=np.float32)
        self.col_index = {col: i for i, col in enumerate(feat_df.columns)}
        
    def step(self, action):
        """Take a step in the environment."""
            # Saving the results
        # self.save_results()
        done = False
        reward = 0

        # record current position
        self.df.loc[self.index, 'trader'] = self.curr_trader

        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]

        # Calculate the value of the asset at time T
        begin_asset = self.calculate_asset()       
         
        # process trade
        reward = self.process_trade(action)
        self.total_reward += reward
        
        #get next obs
        self.index += 1
        observation = self.get_observation()
        
        # Calculate the value of the asset at time T+1
        end_asset = self.calculate_asset()
        
        #record some useful info
        self.record_info(begin_asset, action, reward)

        if self.index >= len(self.df) - 1 or self.balance <= 0:
            done = True
            reward += self.finalize_episode(self.balance)
            self.total_reward += reward
            print(f'total reward : {self.total_reward}')  
            
        # update Contract Price 
        self.update_contract_price()
        
        return observation, reward, done, False, self.get_info(begin_asset)
        
    def process_trade(self, action):
        """Executes a trade based on the action."""
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT + 25
        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy 
            # Long position   
            if self.shares >= 0:
                
                self.balance -= (self.CONTRACT_PRICE + transaction_fee) * action
                self.initial_margin += self.INITIAL_MARGIN * action
                self.maintenance_margin += self.MAINTENANCE_MARGIN * action
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
                self.initial_margin -= self.INITIAL_MARGIN * abs(action)
                self.maintenance_margin -= self.MAINTENANCE_MARGIN * abs(action)            
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
                self.initial_margin -= self.INITIAL_MARGIN * abs(action)
                self.maintenance_margin -= self.MAINTENANCE_MARGIN * abs(action)            
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
                self.initial_margin += self.INITIAL_MARGIN * abs(action)
                self.maintenance_margin += self.MAINTENANCE_MARGIN * abs(action)
            
                # Record the bid price
                for _ in range(abs(action)):
                    # Calculate win rate
                    self.totalRound += 1
                    self.short_trade_round += 1
                    self.last_bid.append(self.df.iloc[self.index]['close']) 
                                   
            self.shares -= abs(action)
        else: # hold
            pass
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
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


    def valid_action_mask_long(self):
        
        actions_mask = np.ones(201, dtype=int)
        if not self.is_tradable():
            actions_mask[0:100] = 0
            actions_mask[101:201] = 0
            return actions_mask.astype(bool)
        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))   
        available_amount = min (5-abs(self.shares), available_amount)
        # long position
        if self.shares >= 0:                   
            # mask invalid action
            actions_mask[101+available_amount:201] = 0
            actions_mask[0:100-self.shares] = 0
        # short position
        elif self.shares < 0:      
            # mask invalid action
            actions_mask[101+abs(self.shares):201] = 0
            actions_mask[0:min(100+abs(self.shares), 201)] = 0  
        return actions_mask.astype(bool)  

    def valid_action_mask_short(self):
        actions_mask = np.ones(201, dtype=int)
        if not self.is_tradable():
            actions_mask[0:100] = 0
            actions_mask[101:201] = 0
            return actions_mask.astype(bool)

        transaction_fee = self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT
        available_amount = int(self.balance // (self.CONTRACT_PRICE + transaction_fee))
        available_amount = min (5-abs(self.shares), available_amount)  
        # long position
        if self.shares > 0:
            # mask invalid action
            actions_mask[101-self.shares:201] = 0
            actions_mask[0:100-self.shares] = 0
        # short position
        elif self.shares <= 0:        
            # mask invalid action
            actions_mask[101+abs(self.shares):201] = 0
            actions_mask[0:100-available_amount] = 0
        return actions_mask.astype(bool)          

    def finalize_episode(self, asset):
        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        reward = 0
        self.balance += self.CONTRACT_PRICE * abs(self.shares)
        self.balance -= (self.df.iloc[self.index]['close'] * self.POINT_VALUE * self.TRANSACTION_FEE_PERCENT) * abs(self.shares)
        self.initial_margin -= self.INITIAL_MARGIN * abs(self.shares)
        self.maintenance_margin -= self.MAINTENANCE_MARGIN * abs(self.shares)  
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
        self.index += 1 
        self.record_info(self.balance, -self.shares, reward)
        self.index -= 1 
        self.df['balance'] = self.feat_arr[:, self.col_index["balance"]]
        self.df['asset'] = self.feat_arr[:, self.col_index["asset"]]
        self.df['shares'] = self.feat_arr[:, self.col_index["shares"]]
        self.df['action'] = self.feat_arr[:, self.col_index["action"]]
        self.df['reward'] = self.feat_arr[:, self.col_index["reward"]]
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        
        if self.totalRound != 0:
            print(f'total win rate : {self.win / self.totalRound}')
        if self.long_trade_round != 0:
            print(f'long trader win rate : {self.long_win / self.long_trade_round}')
        if self.short_trade_round != 0:
            print(f'short trader win rate : {self.short_win / self.short_trade_round}')
        print(f'total trade num : {self.totalRound}')
        print(f'long trader trade num : {self.long_trade_round}')
        print(f'short trader trade num : {self.short_trade_round}')
        
        # Saving the results
        self.save_results()
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
            
        return reward
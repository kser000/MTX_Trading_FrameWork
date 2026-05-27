from .MTXEnvSyn import *
from .MTXEnvBase import *
class MTXEnvLongSimple(MTXEnvSyn):
        
    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0

        # Adjusting the size or intensity of the action
        action = int(action-100) # action : [-100 ~ 100]
        
        # process trade
        reward = self.process_trade(action)
        self.total_reward += reward
        
        #get next obs
        self.index += 1
        observation = self.get_observation()
        
        
        #record some useful info
        self.record_info(self.balance, action, reward)

        if self.index >= len(self.time_arr) - 1:
            done = True
            reward += self.finalize_episode(self.balance)
            self.total_reward += reward
            #print(f'total reward : {self.total_reward}')  
            
        # update Contract Price 
        self.update_contract_price()
        
        return observation, reward, done, False, self.get_info()
                        
    def process_trade(self, action):
        """Executes a trade based on the action."""

        reward = 0
        # Determine whether to buy, sell, or hold
        if action > 0: # buy    
            self.shares += action
            # Record the bid price
            for _ in range(action):
                # Calculate win rate
                self.totalRound += 1
                self.last_bid.append(self.feat_arr[self.index, self.col_index['close']])
        
        elif action < 0: # sell
            for _ in range(abs(action)):
                bid_price = self.last_bid.pop(0)
                reward += self.feat_arr[self.index, self.col_index['close']] - bid_price
                if self.feat_arr[self.index, self.col_index['close']] > bid_price:
                    self.win += 1
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
        asset = self.balance + self.CONTRACT_PRICE * self.shares
        for i in range(int(self.shares)):
            asset +=  self.POINT_VALUE * (self.feat_arr[self.index, self.col_index['close']] - self.last_bid[i])
        
        return asset
    
    def valid_action_mask(self):
        actions_mask = np.ones(201, dtype=bool)
        if not self.is_tradable():
            actions_mask[0:100] = False
            actions_mask[101:201] = False
        elif self.shares <= 0:  # only hold(100) and buy action 1 (action 101) valid
            actions_mask[0:100] = False
            actions_mask[102:201] = False
        elif self.shares > 0:  # only hold(100) and sell action 1 (action 99) valid
            actions_mask[0:99] = False
            actions_mask[101:201] = False
        
        return actions_mask  

    def finalize_episode(self, asset):
        """
        Finalizes the episode, calculates the reward, and updates balances and trades.

        Args:
            asset (float): The total asset value at the previous time step.

        Returns:
            float: The calculated reward for the episode.
        """
        reward = 0
        for i in range(len(self.last_bid)):
            bid_price = self.last_bid.pop(0)
            reward += self.feat_arr[self.index, self.col_index['close']] - bid_price
            if self.feat_arr[self.index, self.col_index['close']] > bid_price:
                self.win += 1
        
        self.index += 1 
        self.record_info(self.balance, -self.shares, reward)
        self.index -= 1 
        '''self.df['balance'] = self.feat_arr[:, self.col_index["balance"]]
        self.df['asset'] = self.feat_arr[:, self.col_index["asset"]]
        self.df['shares'] = self.feat_arr[:, self.col_index["shares"]]
        self.df['action'] = self.feat_arr[:, self.col_index["action"]]
        self.df['reward'] = self.feat_arr[:, self.col_index["reward"]]
        self.df['return'] = self.df['asset'].pct_change(1).fillna(0) * 100
        self.df['cumulative return'] = ((1 + self.df['return'] / 100).cumprod() - 1) * 100
        #print(f'win rate : {self.win / self.totalRound}')
        # Saving the results
        self.save_results()'''
        
        if reward < 0:
            reward = reward * 0.1 * 1.5
        else:
            reward = reward * 0.1
            
        return reward
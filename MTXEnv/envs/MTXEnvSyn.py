from .MTXEnvBase import *

def data_to_numpy(df):
    time_arr = df["date"].values.astype("datetime64[ns]")
    feat_df = df.drop(columns=["date"] + (None or []))
    feat_arr = feat_df.to_numpy(dtype=np.float32)
    col_index = {col: i for i, col in enumerate(feat_df.columns)}
    return time_arr, feat_arr, col_index
class MTXEnvSyn(MTXEnvBase):

    def __init__(self, save_dir, train_start, train_end, sim_start, sim_end, init_balance, feat_columns=None, mode="Train", data_dir="processed_data") -> None:
        super().__init__(save_dir, train_start, train_end, sim_start, sim_end, init_balance, feat_columns, mode, data_dir)
        self.reverse = 0
    def load_entire_data(self):
        """
        Loads the entire training dataset for normalization.

        Returns:
            pd.DataFrame: Concatenated DataFrame of the entire dataset for the specified date range.
        """
        whole_df = pd.DataFrame()
        
        for i in range(2):
            mi = MonthIterator(self.train_start, self.train_end, infinite=False)
            for m in mi:
                if i == 1:
                    file_path = f'{self.data_dir}_reverse_{self.train_end}/mtx-{m}-{self.scalar}min-r.csv'
                    key = f"{m}-r"
                else:
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


    # training reset environ
    def reset_environment(self):
        if self.sim_cur is None:
            self.sim_cur = next(self.month_iter)
            self.reverse = 0
        else:
            self.reverse +=1
            if self.reverse > 1:
                self.reverse = 0
                self.sim_cur = next(self.month_iter)

                   
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
        if self.reverse == 0:
            #file_path = f'processed_data/mtx-{self.sim_cur}-{self.scalar}min.csv'
            key = self.sim_cur
        elif self.reverse == 1:
            #file_path = f'processed_data_reverse_{self.train_end}/mtx-{self.sim_cur}-{self.scalar}min-r.csv'
            key = f"{self.sim_cur}-r"
        self.time_arr = self.time_arr_all[key]
        self.feat_arr = self.feat_arr_all[key]
        self.col_index = self.col_index_all[key]
        self.curr_key = key  # Set current key for pre-normalized data access
        self.position = 'None'
        self.index = self.WINDOW_SIZE-1
        self.balance = self.init_balance
        self.shares = 0
        self.total_reward = 0
        self.last_bid = []
        self.mergin = 0
        
        self.win = 0
        self.totalRound = 0
        if self.sim_cur == self.sim_start:
            self.reset_prices()
        observation = self.get_observation()
        return observation, self.get_info()   
    
    def save_results(self):
        if self.mode != "Train":
            #Saves the simulation results to the file system.
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            if self.reverse == 0:
                save_path = os.path.join(self.save_dir, f'mtx-{self.sim_cur}-{self.scalar}min.csv')
            elif self.reverse == 1:
                save_path = os.path.join(self.save_dir, f'mtx-{self.sim_cur}-{self.scalar}min-r.csv')
            self.df.to_csv(save_path, index=False)
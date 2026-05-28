import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size: [seq_len]；預測 horizon 固定 1 步
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        self.pred_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size: [seq_len]；預測 horizon 固定 1 步
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        self.pred_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_end=None, train_start=None, test_end=None, val_ratio=0.1, cols=None):
        # size: [seq_len]；預測 horizon 固定 1 步
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        self.pred_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path
        self.train_end = train_end
        self.train_start = train_start
        self.test_end = test_end
        self.val_ratio = val_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        cols 若指定：只使用該串列欄位作為資料；未指定時沿用 [date + 其餘欄 + target] 且 cols_data = columns[3:]
        '''
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        if self.cols is not None:
            # 只保留 date 與 cols 中存在的欄位，且確保 target 在內（若不在則自動加入）
            use_cols = [c for c in self.cols if c in df_raw.columns]
            if self.target not in use_cols and self.target in df_raw.columns:
                use_cols.append(self.target)
            cols_to_keep = ['date'] + use_cols
            if 'type' in df_raw.columns and 'type' not in cols_to_keep:
                cols_to_keep = cols_to_keep + ['type']
            df_raw = df_raw[cols_to_keep].copy()
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]

        # 如果提供了 train_end，使用時間劃分；否則使用比例劃分
        if self.train_end is not None:
            # 時間劃分模式（可選 train_start / test_end）
            train_end_dt = pd.to_datetime(self.train_end)
            train_end_mask = df_raw['date'] <= train_end_dt
            if train_end_mask.sum() == 0:
                raise ValueError(f"train_end '{self.train_end}' is earlier than data start time {df_raw['date'].iloc[0]}")
            train_end_idx = df_raw[train_end_mask].index[-1] + 1

            if self.train_start is not None:
                train_start_dt = pd.to_datetime(self.train_start)
                train_start_mask = df_raw['date'] >= train_start_dt
                if train_start_mask.sum() == 0:
                    raise ValueError(f"train_start '{self.train_start}' is later than data end time {df_raw['date'].iloc[-1]}")
                train_start_idx = int(df_raw[train_start_mask].index[0])
            else:
                train_start_idx = 0

            if self.test_end is not None:
                test_end_dt = pd.to_datetime(self.test_end)
                test_end_mask = df_raw['date'] <= test_end_dt
                if test_end_mask.sum() == 0:
                    raise ValueError(f"test_end '{self.test_end}' is earlier than data start time {df_raw['date'].iloc[0]}")
                test_end_idx = int(df_raw[test_end_mask].index[-1] + 1)
                test_end_idx = min(test_end_idx, len(df_raw))
            else:
                test_end_idx = len(df_raw)
            
            # 計算驗證集大小（從 train_start~train_end 區間內劃分）
            train_size = train_end_idx - train_start_idx - self.seq_len
            if train_size <= 0:
                raise ValueError(f"train_start~train_end 區間不足（需至少 seq_len+1 筆），train_start_idx={train_start_idx}, train_end_idx={train_end_idx}")
            val_size = int(train_size * self.val_ratio)
            min_val_size = self.seq_len + self.pred_len
            if val_size < min_val_size:
                val_size = min_val_size
                print(f"Warning: Validation set size adjusted to {val_size} (minimum requirement)")
            
            # 邊界：Train / Val 在 train_start~train_end 內；Test 在 train_end 後到 test_end
            border1s = [
                train_start_idx,
                (train_end_idx - val_size) - self.seq_len,
                train_end_idx - self.seq_len
            ]
            border2s = [
                train_end_idx - val_size,
                train_end_idx,
                test_end_idx
            ]
            
            print(f"Time-based split mode (train_start={self.train_start}, train_end={self.train_end}, test_end={self.test_end})")
            print(f"  Train set: [{df_raw.iloc[border1s[0]]['date']} ~ {df_raw.iloc[border2s[0]-1]['date']}] ({border2s[0] - border1s[0]} rows)")
            print(f"  Validation set: [{df_raw.iloc[border1s[1]]['date']} ~ {df_raw.iloc[border2s[1]-1]['date']}] ({border2s[1] - border1s[1]} rows)")
            if border2s[2] > border1s[2]:
                print(f"  Test set: [{df_raw.iloc[border1s[2]]['date']} ~ {df_raw.iloc[border2s[2]-1]['date']}] ({border2s[2] - border1s[2]} rows)")
            else:
                print(f"  Test set: (empty, border1={border1s[2]}, border2={border2s[2]})")
        else:
            # 原來的比例劃分模式
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            if self.cols is not None:
                cols_data = [c for c in df_raw.columns if c not in ('date', 'type')]
                df_data = df_raw[cols_data]
            else:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # 保存時間信息，用於後續生成 CSV 時添加日期列
        self.dates = df_stamp['date'].values
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # target 在 data_x 的欄位索引（MS 時 __getitem__ 只回傳 target 欄；CSV inverse 用）
        data_x_columns = list(df_data.columns)
        self.target_col_idx = data_x_columns.index(self.target) if self.target in data_x_columns else 0
        if 'type' in df_raw.columns:
            self.session_ids = df_raw['type'].iloc[border1:border2].astype(np.int64).tolist()
        else:
            self.session_ids = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.features == 'MS':
            seq_y = seq_y[:, self.target_col_idx : self.target_col_idx + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        session_id = self.session_ids[r_begin] if self.session_ids is not None else 0
        return seq_x, seq_y, seq_x_mark, seq_y_mark, session_id

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Pretrain(Dataset):
    """
    用于预训练的 Dataset，使用所有数据，不分割 train/val/test
    使用 feature array 来选择特征列，而不是 M/MS/S 分类
    """
    def __init__(self, root_path, data_path='ETTh1.csv',
                 features=None, scale=True, timeenc=0, freq='h', size=None):
        # size: [seq_len]；預測 horizon 固定 1 步
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        self.pred_len = 1
        
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        '''
        df_raw.columns: ['date', ...(features)]
        '''
        # 轉換日期列為 datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 检查所有特征列是否存在
        required_cols = ['date'] + self.features
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Data is missing the following columns: {missing_cols}")
        
        # 使用特征列表选择数据
        df_data = df_raw[self.features]
        
        if self.scale:
            # 使用所有数据拟合 scaler
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data
        self.data_y = data  # 预训练：输入和输出相同（自监督学习）
        self.data_stamp = data_stamp
        
        print(f"Pretrain dataset: Using all {len(self.data_x)} rows")
        print(f"Features: {self.features}")
        print(f"Feature dimension: {len(self.features)}")
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_BinaryTrend(Dataset):
    """
    自定义 Dataset，将 Y 改为 binary trend (二分类趋势)
    - X 保持不变：输入序列 data[s_begin:s_end]
    - Y 直接从数据的 'trend' 列读取：data['trend'][s_end]
    - 如果 separate=True，则 Y 从 'night_trend' 和 'day_trend' 读取
    
    例如：
    - 如果 X 是 data[0:100]，则 Y 是 data['trend'][100]
    - 如果 separate=True，则 Y 是 [data['night_trend'][100], data['day_trend'][100]]
    """
    def __init__(self, root_path, flag='train', size=None,
                 features=None, data_path='ETTh1.csv',
                 target='trend', scaler_path=None, timeenc=0, freq='h',
                 train_end=None, val_end=None, test_end=None, separate=False):
        # size: [seq_len]（分類任務僅需 seq_len）
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scaler_path = scaler_path
        self.separate = separate  # 是否使用夜日盘分开的 trend

        self.root_path = root_path
        self.data_path = data_path
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end
        
        # 确保必须有 train_end 和 val_end
        if self.train_end is None:
            raise ValueError("Dataset_Custom_BinaryTrend must provide train_end parameter!")
        if self.val_end is None:
            raise ValueError("Dataset_Custom_BinaryTrend must provide val_end parameter!")
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature, 'trend']
        注意：数据中必须包含 'trend' 列作为标签
        如果 separate=True，则必须包含 'night_trend' 和 'day_trend' 列
        '''
        # 检查必要的列
        if self.separate:
            # 使用夜日盘分开的 trend
            if 'night_trend' not in df_raw.columns or 'day_trend' not in df_raw.columns:
                raise ValueError("Data must contain 'night_trend' and 'day_trend' columns when separate=True!")
            required_cols = ['date', 'night_trend', 'day_trend'] + self.features
        else:
            # 使用整日 trend
            if 'trend' not in df_raw.columns:
                raise ValueError("Data must contain 'trend' column! Please ensure trend column is added during data preprocessing.")
            required_cols = ['date', 'trend'] + self.features
        
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Data is missing the following columns: {missing_cols}")
        
        # 轉換日期列為 datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 必須使用時間劃分模式（train_end 和 val_end 已在 __init__ 中檢查）
        train_end_dt = pd.to_datetime(self.train_end)
        val_end_dt = pd.to_datetime(self.val_end)
        
        # 找到 train_end 對應的索引（最後一個 <= train_end 的索引 + 1）
        train_end_mask = df_raw['date'] <= train_end_dt
        if train_end_mask.sum() == 0:
            raise ValueError(f"train_end '{self.train_end}' is earlier than data start time {df_raw['date'].iloc[0]}")
        train_end_idx = df_raw[train_end_mask].index[-1] + 1
        
        # 找到 val_end 對應的索引（最後一個 <= val_end 的索引 + 1）
        val_end_mask = df_raw['date'] <= val_end_dt
        if val_end_mask.sum() == 0:
            raise ValueError(f"val_end '{self.val_end}' is earlier than data start time {df_raw['date'].iloc[0]}")
        val_end_idx = df_raw[val_end_mask].index[-1] + 1
        
        # 驗證 val_end > train_end
        if val_end_idx <= train_end_idx:
            raise ValueError(f"val_end '{self.val_end}' must be later than train_end '{self.train_end}'")
        
        # 如果提供了 test_end，使用 test_end；否则使用数据末尾
        if self.test_end is not None:
            test_end_dt = pd.to_datetime(self.test_end)
            test_end_mask = df_raw['date'] <= test_end_dt
            if test_end_mask.sum() == 0:
                raise ValueError(f"test_end '{self.test_end}' is earlier than data start time {df_raw['date'].iloc[0]}")
            test_end_idx = df_raw[test_end_mask].index[-1] + 1
            
            # 驗證 test_end > val_end
            if test_end_idx <= val_end_idx:
                raise ValueError(f"test_end '{self.test_end}' must be later than val_end '{self.val_end}'")
        else:
            test_end_idx = len(df_raw)
        
        # 計算邊界
        # Train: [0, train_end_idx]
        # Val: [train_end_idx - seq_len, val_end_idx]
        # Test: [val_end_idx - seq_len, test_end_idx]
        # 注意：val 和 test 都需要預留 seq_len，確保第一個樣本的 Y 是正確的
        border1s = [
            0,  # Train 開始
            train_end_idx - self.seq_len,  # Val 開始（預留 seq_len）
            val_end_idx - self.seq_len  # Test 開始（預留 seq_len）
        ]
        border2s = [
            train_end_idx,  # Train 結束
            val_end_idx,  # Val 結束
            test_end_idx  # Test 結束
        ]
        
        print(f"Time-based split mode (BinaryTrend):")
        print(f"  train_end_idx: {train_end_idx}")
        print(f"  val_end_idx: {val_end_idx}")
        print(f"  test_end_idx: {test_end_idx}")
        print(f"  Train set: [{df_raw.iloc[border1s[0]]['date']} ~ {df_raw.iloc[border2s[0]-1]['date']}] ({border2s[0] - border1s[0]} rows)")
        print(f"  Validation set: [{df_raw.iloc[border1s[1]]['date']} ~ {df_raw.iloc[border2s[1]-1]['date']}] ({border2s[1] - border1s[1]} rows)")
        print(f"  Test set: [{df_raw.iloc[border1s[2]]['date']} ~ {df_raw.iloc[border2s[2]-1]['date']}] ({border2s[2] - border1s[2]} rows)")
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Debug: Check boundaries and dataset size
        if border2 <= border1:
            raise ValueError(f"Dataset boundary error: border1={border1}, border2={border2}, set_type={self.set_type}")

        # 使用特征列表选择数据
        # 检查所有特征列是否存在
        # 选择指定的特征列
        df_data = df_raw[self.features]

        # 如果提供了 scaler_path，则加载 scaler；否则用原方法 fit
        if self.scaler_path is not None:
            # 从文件加载 scaler
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            data = self.scaler.transform(df_data.values)
        else:
            # 原方法：在 train 数据上 fit
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

        # 保存日期信息（直接使用 dates，不进行时间编码）
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.dates = df_stamp['date'].values

        self.data_x = data[border1:border2]
        
        # 单独保存 trend 列（不进行标准化，因为它是标签）
        if self.separate:
            # 保存 night_trend 和 day_trend
            self.data_night_trend = df_raw['night_trend'].values[border1:border2]
            self.data_day_trend = df_raw['day_trend'].values[border1:border2]
        else:
            # 保存整日 trend
            self.data_trend = df_raw[self.target].values[border1:border2]
        
        # Debug: Print dataset size
        type_map = {'train': 0, 'val': 1, 'test': 2}
        flag_name = [k for k, v in type_map.items() if v == self.set_type][0]
        dataset_len = max(0, len(self.data_x) - self.seq_len)
        if self.separate:
            print(f"  {flag_name} dataset size: data_x={len(self.data_x)}, night_trend={len(self.data_night_trend)}, day_trend={len(self.data_day_trend)}, __len__={dataset_len}")
        else:
            print(f"  {flag_name} dataset size: data_x={len(self.data_x)}, data_trend={len(self.data_trend)}, __len__={dataset_len}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        # X 保持不变：输入序列 data[s_begin:s_end]
        seq_x = self.data_x[s_begin:s_end]
        
        # Y 根据 separate 参数选择
        if self.separate:
            # 使用 night_trend 和 day_trend
            # 返回两个独立的二分类标签：[night_trend, day_trend]
            night_value = float(self.data_night_trend[s_end])
            day_value = float(self.data_day_trend[s_end])
            seq_y = np.array([night_value, day_value], dtype=np.float32)  # [2]
        else:
            # 使用整日 trend
            trend_value = self.data_trend[s_end]
            seq_y = np.array([float(trend_value)], dtype=np.float32)  # [1]

        return seq_x, seq_y

    def __len__(self):
        # 由于 Y 是 data['trend'][s_end]，需要确保 s_end 在范围内
        # s_end = index + seq_len，所以 index 最大为 len(self.data_x) - seq_len - 1
        # 但为了确保 s_end 在 data_trend 范围内，需要确保 s_end < len(self.data_trend)
        # 由于 len(self.data_trend) == len(self.data_x)，所以返回 len(self.data_x) - self.seq_len
        # 但要确保至少返回 0（如果数据不足）
        return max(0, len(self.data_x) - self.seq_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    """
    預測用資料集；輸出 horizon 固定 1 步（與訓練一致）。
    - 有 pred_start/pred_end：只載入 [pred_start 前 seq_len 筆, pred_end] 列，每個範圍內錨點一筆樣本。
    - 否則：滑動視窗 + 時間戳末尾接 1 個合成未來日期（供 seq_y_mark）。
    """

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 scaler_path=None, pred_start=None, pred_end=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_path = scaler_path
        self.pred_start = pred_start
        self.pred_end = pred_end
        self.__read_data__()

    def __read_data__(self):
        import pickle
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.reset_index(drop=True)

        self._range_mode = False
        self._extend_stamp = False

        if self.pred_start is not None or self.pred_end is not None:
            pred_start_date = pd.to_datetime(self.pred_start) if self.pred_start is not None else df_raw['date'].iloc[-1]
            pred_end_date = pd.to_datetime(self.pred_end) if self.pred_end is not None else df_raw['date'].iloc[-1]
            # range_mode: 後續用 border1/border2 直接切出錨點日期/盤別
            self._range_mode = True
            self.pred_dates_list = None
            self.pred_type_list = None
        else:
            # no-range mode：預測「下一個 session」
            # - 若最後一筆是 night(type=0) -> 下一筆是同日 day(type=1)
            # - 若最後一筆是 day(type=1)   -> 下一筆是隔日 night(type=0)
            last_date = df_raw['date'].iloc[-1]
            last_sid = int(df_raw['type'].iloc[-1])
            self.default_pred_session_id = 1 - last_sid
            next_sid = int(self.default_pred_session_id)
            next_date = last_date if (last_sid == 0 and next_sid == 1) else (last_date + pd.Timedelta(days=1))
            self.pred_dates_list = [next_date]
            self.pred_type_list = [next_sid]

        use_cols = [c for c in self.cols if c in df_raw.columns]
        if self.target not in use_cols and self.target in df_raw.columns:
            use_cols.append(self.target)
        cols_to_keep = ['date'] + use_cols
        if 'type' in df_raw.columns and 'type' not in cols_to_keep:
            cols_to_keep.append('type')
        df_raw = df_raw[cols_to_keep].copy()

        self.df_raw_full = df_raw.copy()

        if self.features == 'M' or self.features == 'MS':
            cols_data = [c for c in df_raw.columns if c not in ('date', 'type')]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.scaler_path and os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Scaler loaded from: {self.scaler_path}")
            else:
                self.scaler.fit(df_data.values)
                print("Scaler fitted from prediction data")
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        n = len(data)
        border1, border2 = 0, n

        if self._range_mode:
            i0 = int(df_raw['date'].searchsorted(pred_start_date, side='left'))
            i1 = int(df_raw['date'].searchsorted(pred_end_date, side='right'))
            # caller assumes pred_start/pred_end are valid (sufficient history exists)
            border1 = i0 - self.seq_len
            border2 = i1
            # Dataset 的樣本順序：index=0 -> r_begin=seq_len (anchor rows)
            anchor_start = border1 + self.seq_len  # == i0
            anchor_end = border2
            self.pred_dates_list = df_raw['date'].iloc[anchor_start:anchor_end].values
            if 'type' in df_raw.columns:
                self.pred_type_list = df_raw['type'].iloc[anchor_start:anchor_end].astype(np.int64).values
            else:
                self.pred_type_list = None
        else:
            self._extend_stamp = True

        df_stamp = pd.DataFrame(columns=['date'])
        if self._extend_stamp:
            tmp_stamp = df_raw[['date']][border1:border2]
            tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'])
            df_stamp['date'] = list(tmp_stamp['date'].values) + self.pred_dates_list
        else:
            tmp_stamp = df_raw[['date']][border1:border2]
            tmp_stamp['date'] = pd.to_datetime(tmp_stamp['date'])
            df_stamp['date'] = tmp_stamp['date'].values

        self.df_raw_dates = df_raw['date'].values

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if 'type' in df_raw.columns:
            self.session_ids = df_raw['type'].iloc[border1:border2].astype(np.int64).tolist()
        else:
            self.session_ids = None

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        data_x_columns = list(df_data.columns)
        self.target_col_idx = data_x_columns.index(self.target) if self.target in data_x_columns else 0

        self.pred_ground_truth_scaled = None
        if self._range_mode:
            tci = self.target_col_idx
            # anchor rows are exactly data_y[self.seq_len:] when pred_start/pred_end covers a contiguous block
            # return as numpy 1d array (exp_main 只需要 pred_len=1 的標量真值)
            self.pred_ground_truth_scaled = self.data_y[self.seq_len:, tci].astype(np.float64)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + 1

        seq_x = self.data_x[s_begin:s_end]
        if self._range_mode:
            n_feat = 1 if self.features == 'MS' else self.data_y.shape[1]
            seq_y = np.zeros((1, n_feat), dtype=self.data_y.dtype)
        else:
            # no-range mode：允許輸入剛好 seq_len 列（dataset len=1）
            # 這時 r_begin == len(data_y)，取不到真值，回傳 0 占位即可（exp_main.predict 會忽略 batch_y）
            if r_begin >= len(self.data_y):
                n_feat = 1 if self.features == 'MS' else self.data_y.shape[1]
                seq_y = np.zeros((1, n_feat), dtype=self.data_y.dtype)
            elif self.inverse:
                seq_y = self.data_x[r_begin:r_end]
            else:
                seq_y = self.data_y[r_begin:r_end]
        if self.features == 'MS' and not self._range_mode:
            seq_y = seq_y[:, self.target_col_idx : self.target_col_idx + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        if self.session_ids is not None and r_begin < len(self.session_ids):
            session_id = int(self.session_ids[r_begin])
        else:
            session_id = int(getattr(self, 'default_pred_session_id', 0))
        return seq_x, seq_y, seq_x_mark, seq_y_mark, session_id

    def __len__(self):
        if self._range_mode:
            return max(0, len(self.data_x) - self.seq_len)
        # no-range mode：若剛好只有 seq_len 列，也要能回傳 1 筆樣本（預測下一個 session）
        n = len(self.data_x)
        if self._extend_stamp and n == self.seq_len:
            return 1
        return max(0, n - self.seq_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _parse_cols_arg(cols):
    if cols is None:
        return None
    if isinstance(cols, (list, tuple)):
        return list(cols)
    s = str(cols)
    return [c.strip() for c in s.replace(",", " ").split() if c.strip()]


def _find_date_index(dates_array, target_date):
    """Return first index where dates_array == target_date (normalized to date)."""
    td = pd.to_datetime(target_date).normalize()
    da = pd.to_datetime(pd.Series(dates_array)).dt.normalize().to_numpy()
    hit = np.where(da == td)[0]
    return int(hit[0]) if len(hit) else None


if __name__ == "__main__":
    # Debug harness: compare Dataset_Custom(test) vs Dataset_Pred for same anchor-date X window.
    import argparse
    import tempfile

    p = argparse.ArgumentParser(description="Compare Dataset_Custom(test) vs Dataset_Pred X windows for same y-anchor date")
    p.add_argument("--root_path", default="../dataset/", help="Dataset folder (same as scripts/PatchTST/mtx_spot_separate.sh)")
    p.add_argument("--data_path", default="MTX_separate_sessions.csv")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--features", default="MS")
    p.add_argument("--target", default="log_return")
    p.add_argument("--freq", default="d")
    p.add_argument("--cols", default="log_return rolling_std macd mfi cci adx n225 sp500 twse")
    p.add_argument("--train_start", default="2019-01-01")
    p.add_argument("--train_end", default="2025-12-31")
    p.add_argument("--test_end", default="2026-12-31")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--pred_start", default="2026-01-01")
    p.add_argument("--pred_end", default="2026-03-18")
    p.add_argument("--anchor_date", default=None, help="Optional: specific y-anchor date (YYYY-MM-DD). If None, use pred_start.")
    p.add_argument("--scaler_path", default="checkpoints/MTX_spot_separate_64_2026/scaler.pkl", help="Path to saved scaler file (.pkl). If None, will fit scaler on train data")
    args = p.parse_args()

    cols = _parse_cols_arg(args.cols)
    size = [args.seq_len]

    # 1) Build Dataset_Custom(test) to get the training-fitted scaler.
    ds_test = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag="test",
        size=size,
        features=args.features,
        target=args.target,
        timeenc=0,
        freq=args.freq,
        train_start=args.train_start,
        train_end=args.train_end,
        test_end=args.test_end,
        val_ratio=args.val_ratio,
        cols=cols,
    )
    # 2) Persist scaler to temp and pass to Dataset_Pred (so scale matches test/train).
    ds_pred = Dataset_Pred(
        root_path=args.root_path,
        data_path=args.data_path,
        size=size,
        features=args.features,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=0,
        freq=args.freq,
        cols=cols,
        scaler_path=args.scaler_path,
        pred_start=args.pred_start,
        pred_end=args.pred_end,
    )
        # Choose anchor date (y-date). In this codebase, pred_dates_list is the anchor-date list.
    anchor_date = args.anchor_date or args.pred_start
    pred_i = _find_date_index(getattr(ds_pred, "pred_dates_list", []), anchor_date)
    if pred_i is None:
        raise SystemExit(f"anchor_date {anchor_date} not found in ds_pred.pred_dates_list")

    # Pred X
    pred_x = ds_pred[pred_i][0]

    # Custom(test) X for same anchor date:
    # In Dataset_Custom, y-anchor date corresponds to r_begin = index+seq_len; so we find position of anchor_date in ds_test.dates
    pos = _find_date_index(getattr(ds_test, "dates", []), anchor_date)
    if pos is None:
        raise SystemExit(f"anchor_date {anchor_date} not found in ds_test.dates")
    test_i = pos - args.seq_len
    if test_i < 0:
        raise SystemExit(f"Not enough history in test split for anchor_date={anchor_date} (pos={pos}, seq_len={args.seq_len})")
    test_x = ds_test[test_i][0]

    # Compare
    diff = np.asarray(pred_x) - np.asarray(test_x)
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    ok = bool(np.allclose(pred_x, test_x, rtol=args.rtol, atol=args.atol, equal_nan=True))
    print(f"[compare] anchor_date={pd.to_datetime(anchor_date).date()} pred_i={pred_i} test_i={test_i}")
    print(f"[compare] X shape pred={np.asarray(pred_x).shape} test={np.asarray(test_x).shape}")
    print(f"[compare] allclose={ok} max_abs_diff={max_abs}")

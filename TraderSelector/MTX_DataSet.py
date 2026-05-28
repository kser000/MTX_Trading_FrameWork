import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MTX_Dataset(Dataset):
    def __init__(self, seq_length, start_date, end_date, scaler_norm=None, data_path="TradeSelector_processed_data/denoised_data.csv", features=None, label='trend_denoise'):
        
        # Filter the data within the specified time range.
        df = pd.read_csv(data_path)
        start_idx = df.index[df['date'] >= start_date][0] if len(df.index[df['date'] >= start_date]) > 0 else None
        end_idx = df.index[df['date'] <= end_date][-1] if len(df.index[df['date'] <= end_date]) > 0 else None
        
        if start_idx is not None and end_idx is not None:
            # 確保 start_idx 之前有至少 seq_length 筆數據
            start_idx = max(0, start_idx - seq_length)

            # 先從 start_idx 開始切片，然後篩選出符合日期範圍的資料
            filtered_df = df.iloc[start_idx:end_idx + 1]
        else:
            # 若 start_date 或 end_date 不在 df 的範圍內，返回空 DataFrame
            filtered_df = pd.DataFrame(columns=df.columns)
        
        # 設定預設特徵
        if features is None:
            features = ['close_denoised']
        
        # 檢查特徵是否存在於數據中
        available_features = [f for f in features if f in filtered_df.columns]
        if len(available_features) != len(features):
            missing_features = set(features) - set(available_features)
            print(f"Warning: Missing features {missing_features}. Using available features: {available_features}")
        
        self.features = available_features
        self.feature_count = len(available_features)
        
        # 根據選擇的特徵載入數據
        if len(available_features) == 1:
            # 單一特徵
            self.data = torch.tensor(filtered_df[available_features[0]].values, dtype=torch.float32)
        else:
            # 多特徵
            self.data = torch.tensor(filtered_df[available_features].values, dtype=torch.float32)

        # 如果未提供scaler，則訓練集上進行正規化，並保存scaler
        # 改用 StandardScaler 來處理技術指標（更適合不同尺度的特徵）
        if scaler_norm is None:
            self.normalized_data, self.scaler_norm = self.normalise_data_xlstm(self.data)
        else:
            # 若提供scaler，則使用已有的scaler來轉換數據
            if len(available_features) == 1:
                self.normalized_data = scaler_norm.transform(self.data.reshape(-1, 1))
            else:
                self.normalized_data = scaler_norm.transform(self.data)
            self.scaler_norm = scaler_norm

        
        self.labels = torch.tensor(filtered_df[label].values, dtype=torch.float32).unsqueeze(1)  # 轉換為 tensor
        self.seq_length = seq_length
        
        
        trend_counts = filtered_df.iloc[seq_length:][label].value_counts()
        print(f'shape of data : {self.data.shape}')
        print(f'shape of labels : {self.labels.shape}')
        print(trend_counts)
        
    def __len__(self):
        return len(self.normalized_data) - self.seq_length

    def __getitem__(self, idx):
        return torch.tensor(self.normalized_data[idx:idx+self.seq_length], dtype=torch.float32), self.labels[idx+self.seq_length]
    
    def normalise_data_xlstm(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 根據數據維度決定是否需要reshape
        if len(data.shape) == 1:
            return scaler.fit_transform(data.reshape(-1, 1)), scaler
        else:
            return scaler.fit_transform(data), scaler   
    
    def standardize_data_xlstm(self, data):

        standard_scaler = StandardScaler()
        
        # 根據數據維度決定是否需要reshape
        if len(data.shape) == 1:
            standardized_data = standard_scaler.fit_transform(data.reshape(-1, 1))
        else:
            standardized_data = standard_scaler.fit_transform(data)
        
        return standardized_data, standard_scaler   

        
if __name__ == "__main__":
    train_dataset = MTX_Dataset(100, '2018-01-04', '2024-06-19', data_path="TradeSelector_processed_data/2025H1_8D_binary.csv", )
    #val_dataset = MTX_Dataset(100, '2020-6-19', '2020-12-16',scaler_norm=train_dataset.scaler_norm)
    #test_dataset = MTX_Dataset(100, '2020-12-17', '2021-6-16', scaler_norm=train_dataset.scaler_norm)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # validation/test 不用 shuffle
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # plot_density(train_dataset.normalized_data, test_dataset.normalized_data, val_dataset.normalized_data)

        
    for batch_x, batch_y in train_loader:
        print(f'shape of batch x : {batch_x.shape}')
        print(f'{batch_x}')
        print(f'shape of batch y : {batch_y.shape}')
        print(f'{batch_y}')
        break
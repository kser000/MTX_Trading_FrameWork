"""
MTX Dataset for Regression Task
預測隔天的絕對價格
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MTX_Dataset_Regression(Dataset):
    """
    用於回歸任務的 MTX Dataset - 預測絕對價格
    
    Args:
        seq_length: 序列長度
        start_date: 開始日期
        end_date: 結束日期
        data_path: 數據文件路徑
        features: 特徵列表
        target: 預測目標 ('close_denoised' 或 'close')
        scaler: MinMaxScaler 用於 val/test
    
    Note:
        預測目標：price[t+seq_length]（絕對價格）
    """
    def __init__(self, seq_length, start_date, end_date, 
                 data_path,
                 features=['close_denoised'],
                 scaler=None):
        # 僅允許單一特徵
        assert len(features) == 1, "只允許一個特徵（feature）!"
        feature = features[0]
        # 讀取數據
        df = pd.read_csv(data_path)
        start_idx = df.index[df['date'] >= start_date][0] if len(df.index[df['date'] >= start_date]) > 0 else None
        end_idx = df.index[df['date'] <= end_date][-1] if len(df.index[df['date'] <= end_date]) > 0 else None
        if start_idx is not None and end_idx is not None:
            start_idx = max(0, start_idx - seq_length)
            filtered_df = df.iloc[start_idx:end_idx + 1]
        else:
            filtered_df = pd.DataFrame(columns=df.columns)
        # 單一特徵欄位（特徵與目標同欄位）
        self.data = filtered_df[feature].values.reshape(-1, 1)
        self.feature_count = 1
        # 標準化
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.normalized_data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.normalized_data = self.scaler.transform(self.data)
        # PyTorch tensor 轉換
        self.labels = torch.tensor(self.normalized_data, dtype=torch.float32)
        self.raw_labels = torch.tensor(filtered_df['close'].values, dtype=torch.float32)
        self.seq_length = seq_length
        print(f'Data shape: {self.data.shape}')
        print(f'Feature used (same for target): {feature}')
    def __len__(self):
        return len(self.normalized_data) - self.seq_length
    def __getitem__(self, idx):
        X = torch.tensor(self.normalized_data[idx:idx+self.seq_length], dtype=torch.float32)
        y = self.labels[idx+self.seq_length]
        return X, y
    def denormalize(self, normalized_value):
        if isinstance(normalized_value, torch.Tensor):
            normalized_value = normalized_value.cpu().numpy()
        if normalized_value.ndim == 1:
            normalized_value = normalized_value.reshape(-1, 1)
        return self.scaler.inverse_transform(normalized_value)


class MTX_Dataset_Regression_V2(Dataset):
    """
    用於回歸任務的 MTX Dataset - X 使用去噪數據，Y 使用原始數據
    
    Args:
        seq_length: 序列長度
        start_date: 開始日期
        end_date: 結束日期
        data_path: 數據文件路徑
        x_feature: X 使用的特徵（例如 'close_denoised'）
        y_feature: Y 使用的特徵（例如 'close'，原始數據）
        x_scaler: X 的 MinMaxScaler（用於 val/test）
        y_scaler: Y 的 MinMaxScaler（用於 val/test）
    
    Note:
        - X（輸入）：使用去噪後的數據（denoised data），用 x_scaler 標準化
        - Y（標籤）：使用原始數據（raw data），用 y_scaler 標準化
        - 需要兩個獨立的 scaler，因為 denoised 和 raw 的數值分佈可能不同
    """
    def __init__(self, seq_length, start_date, end_date, 
                 data_path,
                 x_feature='close_denoised',
                 y_feature='close',
                 x_scaler=None,
                 y_scaler=None):
        # 讀取數據
        df = pd.read_csv(data_path)
        start_idx = df.index[df['date'] >= start_date][0] if len(df.index[df['date'] >= start_date]) > 0 else None
        end_idx = df.index[df['date'] <= end_date][-1] if len(df.index[df['date'] <= end_date]) > 0 else None
        if start_idx is not None and end_idx is not None:
            start_idx = max(0, start_idx - seq_length)
            filtered_df = df.iloc[start_idx:end_idx + 1]
        else:
            filtered_df = pd.DataFrame(columns=df.columns)
        
        # 檢查特徵是否存在
        if x_feature not in filtered_df.columns:
            raise ValueError(f"X feature '{x_feature}' not found in data")
        if y_feature not in filtered_df.columns:
            raise ValueError(f"Y feature '{y_feature}' not found in data")
        
        # 提取 X 和 Y 數據
        self.x_data = filtered_df[x_feature].values.reshape(-1, 1)  # X: 去噪數據
        self.y_data = filtered_df[y_feature].values.reshape(-1, 1)  # Y: 原始數據
        self.feature_count = 1
        
        # 標準化 X（去噪數據）
        if x_scaler is None:
            self.x_scaler = StandardScaler()
            self.normalized_x_data = self.x_scaler.fit_transform(self.x_data)
        else:
            self.x_scaler = x_scaler
            self.normalized_x_data = self.x_scaler.transform(self.x_data)
        
        # 標準化 Y（原始數據）
        if y_scaler is None:
            self.y_scaler = StandardScaler()
            self.normalized_y_data = self.y_scaler.fit_transform(self.y_data)
        else:
            self.y_scaler = y_scaler
            self.normalized_y_data = self.y_scaler.transform(self.y_data)
        
        # PyTorch tensor 轉換
        self.labels = torch.tensor(self.normalized_y_data, dtype=torch.float32)
        self.seq_length = seq_length
        
        print(f'X data shape: {self.x_data.shape} (feature: {x_feature})')
        print(f'Y data shape: {self.y_data.shape} (feature: {y_feature})')
        print(f'X range: [{self.x_data.min():.2f}, {self.x_data.max():.2f}]')
        print(f'Y range: [{self.y_data.min():.2f}, {self.y_data.max():.2f}]')
    
    def __len__(self):
        return len(self.normalized_x_data) - self.seq_length
    
    def __getitem__(self, idx):
        # X: 使用去噪數據的標準化序列
        X = torch.tensor(self.normalized_x_data[idx:idx+self.seq_length], dtype=torch.float32)
        # Y: 使用原始數據的標準化值
        y = self.labels[idx+self.seq_length]
        return X, y
    
    def denormalize_y(self, normalized_value):
        """
        反標準化 Y（從標準化值回到原始價格）
        
        Args:
            normalized_value: 標準化的 Y 值（可以是 numpy array 或 torch tensor）
        
        Returns:
            反標準化後的原始價格
        """
        if isinstance(normalized_value, torch.Tensor):
            normalized_value = normalized_value.cpu().numpy()
        if normalized_value.ndim == 1:
            normalized_value = normalized_value.reshape(-1, 1)
        return self.y_scaler.inverse_transform(normalized_value)
    
    def denormalize_x(self, normalized_value):
        """
        反標準化 X（從標準化值回到去噪價格）
        
        Args:
            normalized_value: 標準化的 X 值（可以是 numpy array 或 torch tensor）
        
        Returns:
            反標準化後的去噪價格
        """
        if isinstance(normalized_value, torch.Tensor):
            normalized_value = normalized_value.cpu().numpy()
        if normalized_value.ndim == 1:
            normalized_value = normalized_value.reshape(-1, 1)
        return self.x_scaler.inverse_transform(normalized_value)


class MTX_Dataset_Regression_MultiFeature(Dataset):
    """
    用於回歸任務的 MTX Dataset - 支持多維輸入，label 使用 close_denoised
    
    Args:
        seq_length: 序列長度
        start_date: 開始日期
        end_date: 結束日期
        data_path: 數據文件路徑
        features: 特徵列表（可以多個，例如 ['close_denoised', 'open_denoised', 'high_denoised', 'low_denoised']）
        label: 預測目標（默認 'close_denoised'，必須在 features 中）
        trend_col: trend 列名（默認 'trend_denoise'，如果不存在則嘗試 'trend_raw'）
        scaler: MinMaxScaler 用於 val/test（多維特徵使用 MinMaxScaler）
    
    Note:
        - 輸入特徵：多維特徵（每個特徵獨立標準化）
        - 預測目標：label 必須是 features 中的一個，使用相同的 scaler
        - Trend：額外記錄 trend 信息，可在訓練時使用
    """
    def __init__(self, seq_length, start_date, end_date, 
                 data_path,
                 features=['close_denoised'],
                 label='close_denoised',
                 trend_col='trend_raw',
                 open_col='open',
                 scaler=None):
        # 讀取數據
        df = pd.read_csv(data_path)
        start_idx = df.index[df['date'] >= start_date][0] if len(df.index[df['date'] >= start_date]) > 0 else None
        end_idx = df.index[df['date'] <= end_date][-1] if len(df.index[df['date'] <= end_date]) > 0 else None
        
        if start_idx is not None and end_idx is not None:
            start_idx = max(0, start_idx - seq_length)
            filtered_df = df.iloc[start_idx:end_idx + 1]
        else:
            filtered_df = pd.DataFrame(columns=df.columns)
        
        # 檢查特徵是否存在
        available_features = [f for f in features if f in filtered_df.columns]
        if len(available_features) != len(features):
            missing_features = set(features) - set(available_features)
            print(f"Warning: Missing features {missing_features}. Using available features: {available_features}")
        self.features = available_features
        self.feature_count = len(available_features)
        
        # 檢查 label 是否存在且在 features 中
        if label not in filtered_df.columns:
            raise ValueError(f"Label column '{label}' not found in data")
        if label not in available_features:
            raise ValueError(f"Label '{label}' must be in features list: {available_features}")
        self.label = label
        
        # 找到 label 在 features 中的索引
        self.label_idx = available_features.index(label)
        
        # 提取多維特徵數據
        self.data = filtered_df[available_features].values  # Shape: [N, num_features]
        
        # 標準化特徵（多維特徵使用 MinMaxScaler）
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.normalized_data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.normalized_data = self.scaler.transform(self.data)
        
        # Label 就是 normalized_data 中對應 label 的那一列
        # 不需要單獨的 label_scaler，因為 label 是 features 的一部分
        self.normalized_labels = self.normalized_data[:, self.label_idx].reshape(-1, 1)  # Shape: [N, 1]
        
        # PyTorch tensor 轉換
        self.labels = torch.tensor(self.normalized_labels, dtype=torch.float32)
        self.seq_length = seq_length
        
        # 提取 trend 數據（如果存在）
        self.trend = torch.tensor(filtered_df[trend_col].values, dtype=torch.long)
        print(f'Trend column used: {trend_col}')
        
        # 提取當天的 open（原始值和標準化值）
        self.open_raw = filtered_df[open_col].values  # 原始值（未標準化），用於反標準化後計算 accuracy
        
        open_idx = available_features.index(open_col)
        self.open_normalized = self.normalized_data[:, open_idx]  # 標準化值（用於 DirectionalMSELoss）
        print(f'Data shape: {self.data.shape}')
        print(f'Features used: {available_features}')
        print(f'Label used: {label} (index: {self.label_idx})')
        print(f'Feature count: {self.feature_count}')
    
    def __len__(self):
        return len(self.normalized_data) - self.seq_length
    
    def __getitem__(self, idx):
        # X: [seq_length, num_features]
        X = torch.tensor(self.normalized_data[idx:idx+self.seq_length], dtype=torch.float32)
        # y: [1] (normalized label)
        y = self.labels[idx+self.seq_length]
        
        # trend: 當天的趨勢
        trend = self.trend[idx+self.seq_length]
        
        # open_raw: 當天的 open（原始值，未標準化），用於反標準化後計算 accuracy
        open_raw = self.open_raw[idx+self.seq_length]
        
        # open_normalized: 當天的 open（標準化值），用於 DirectionalMSELoss
        open_normalized = self.open_normalized[idx+self.seq_length]
        
        return X, y, trend, open_raw, open_normalized
    
    def denormalize(self, normalized_value):
        """
        反標準化 label（從標準化值回到絕對價格）
        
        Args:
            normalized_value: 標準化的 label 值（可以是 numpy array 或 torch tensor）
        
        Returns:
            反標準化後的絕對價格
        
        Note:
            label 是 features 的一部分，使用 scaler 的 data_min_ 和 data_max_ 來反標準化
            MinMaxScaler 公式: X_scaled = (X - data_min_) / (data_max_ - data_min_)
            反標準化: X = X_scaled * (data_max_ - data_min_) + data_min_
        """
        if isinstance(normalized_value, torch.Tensor):
            normalized_value = normalized_value.cpu().numpy()
        if normalized_value.ndim == 1:
            normalized_value = normalized_value.reshape(-1, 1)
        
        # 獲取 label 對應特徵的 min 和 max
        data_min = self.scaler.data_min_[self.label_idx]
        data_max = self.scaler.data_max_[self.label_idx]
        
        # 反標準化: X = X_scaled * (data_max - data_min) + data_min
        # 對於 feature_range=(0,1)，X_scaled 已經是 (X - data_min) / (data_max - data_min)
        denormalized = normalized_value * (data_max - data_min) + data_min
        
        return denormalized


if __name__ == "__main__":
    # 測試絕對價格模式（單一特徵）
    
    train_dataset_multi = MTX_Dataset_Regression_MultiFeature(
        seq_length=50,
        start_date='2020-01-01',
        end_date='2020-12-31',
        data_path='TradeSelector_processed_data/regression/2022H1.csv',
        features=['close_denoised', 'open_denoised', 'high_denoised', 'low_denoised'],
        label='close_denoised'
    )
    
    print(f"\nDataset size: {len(train_dataset_multi)}")
    
    # 測試一個樣本
    X, y, trend, open_raw, open_normalized = train_dataset_multi[0]
    print(f"\nSample:")
    print(f"  X shape: {X.shape} (seq_length, num_features)")
    print(f"  y shape: {y.shape}")
    print(f"  y value (normalized): {y.item():.4f}")
    print(f"  trend: {trend}")
    print(f"  open_raw: {open_raw}")
    print(f"  open_normalized: {open_normalized}")
    
    # 反標準化
    y_denorm = train_dataset_multi.denormalize(y)
    print(f"  y value (price): {y_denorm[0][0]:.2f}")
    
    # 測試 scaler 結構
    print(f"  Scaler type: {type(train_dataset_multi.scaler)}")
    
    # 創建 DataLoader
    train_loader_multi = DataLoader(train_dataset_multi, batch_size=32, shuffle=True)
    
    for batch_x, batch_y, trend, open_raw, open_normalized in train_loader_multi:
        print(f"\nBatch:")
        print(f"  X shape: {batch_x.shape} (batch_size, seq_length, num_features)")
        print(f"  y shape: {batch_y.shape} (batch_size, 1)")
        print(f"  open_normalized shape: {open_normalized.shape} (batch_size,)")
        break
    
    # 測試使用 train scaler 創建 test dataset
    print("\n" + "="*80)
    print("Testing Test Dataset with Train Scaler")
    print("="*80)



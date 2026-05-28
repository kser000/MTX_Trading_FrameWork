"""
xLSTM 回歸訓練腳本
預測隔天的去噪收盤價（連續值）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import yaml
import random
from argparse import ArgumentParser
from loss_functions import *
from xLSTM_TS import *
#from xLSTM_TS_improved import create_xlstm_model_improved, xLSTMClassifierImproved
from MTX_DataSet_regression import MTX_Dataset_Regression, MTX_Dataset_Regression_MultiFeature, MTX_Dataset_Regression_V2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import wavelet_denoising


def set_random_seed(seed=42):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_sampler(dataset):
    """
    創建基於趨勢的平衡採樣器（讓上升和下降的頻率一致）
    
    判斷 trend 的方法：label - 最後一個data（標準化後的值）
    trend = 1 if (label - X[-1]) > 0 else 0
    
    Args:
        dataset: MTX_Dataset_Regression 實例
    
    Returns:
        WeightedRandomSampler: 平衡採樣器，如果數據已平衡則返回 None
    
    Note:
        - trend 只有兩類：1 (上升) 或 0 (下降)
        - 計算權重讓上升和下降樣本被採樣的頻率一致
    """
    dataset_size = len(dataset)
    
    # 收集所有樣本的趨勢標籤
    trend_labels = []
    for idx in range(dataset_size):
        X, y, _ ,_, _= dataset[idx]  # X: [seq_length, 1], y: [1], open
        # 判斷 trend: label - 最後一個data
        last_data = X[-1, 1].item()  # 最後一個數據點（標準化後）
        label = y[0].item()  # label（標準化後）
        trend = 1 if (label - last_data) > 0 else 0
        trend_labels.append(trend)
    
    trend_labels = np.array(trend_labels)
    
    # 統計趨勢分佈（trend: 1=上升, 0=下降）
    up_count = (trend_labels == 1).sum()
    down_count = (trend_labels == 0).sum()
    total = len(trend_labels)
    
    print(f"\n{'='*60}")
    print(f"Trend Distribution Analysis")
    print(f"{'='*60}")
    print(f"  Up (1):   {up_count:5d} ({up_count/total*100:5.2f}%)")
    print(f"  Down (0): {down_count:5d} ({down_count/total*100:5.2f}%)")
    print(f"  Total:    {total:5d}")
    
    # 如果數據已經平衡（差異小於 5%），不需要 sampler
    imbalance_ratio = abs(up_count - down_count) / total
    if imbalance_ratio < 0.05:
        print(f"\nData is already balanced (imbalance < 5%), no sampler needed.")
        return None
    
    # 計算權重：讓上升和下降樣本被採樣的頻率一致
    up_weight = total / (2 * up_count) if up_count > 0 else 0
    down_weight = total / (2 * down_count) if down_count > 0 else 0
    
    sample_weights = np.zeros(dataset_size)
    sample_weights[trend_labels == 1] = up_weight
    sample_weights[trend_labels == 0] = down_weight
    
    print(f"\nSample Weights:")
    print(f"  Up weight:   {up_weight:.4f}")
    print(f"  Down weight: {down_weight:.4f}")
    print(f"{'='*60}\n")
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=dataset_size,
        replacement=True
    )
    return sampler


def create_dataloader_regression(seq_length, config_path, data_path):
    """
    創建回歸任務的 DataLoader
    Args:
        seq_length: 序列長度
        config_path: 配置文件路徑
        data_path: 數據路徑
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_start = config['dataset']['train']['start_date']
    train_end = config['dataset']['train']['end_date']
    val_start = config['dataset']['val']['start_date']
    val_end = config['dataset']['val']['end_date']
    test_start = config['dataset']['test']['start_date']
    test_end = config['dataset']['test']['end_date']
    batch_size = config['dataloader'].get('batch_size', 32)
    #features = config['model'].get('features', ['close_denoised'])
    x_feature = 'close_denoised'
    y_feature = 'close'
    print(f"train start date : {train_start}, train end date : {train_end}")
    print(f"test start date : {test_start}, test end date : {test_end}")
    print(f"\nCreating Train Dataset...")
    train_dataset = MTX_Dataset_Regression_V2(
        seq_length, train_start, train_end,
        data_path=data_path,
        x_feature=x_feature,
        y_feature=y_feature,
        x_scaler=None,
        y_scaler=None
    )
    val_dataset = MTX_Dataset_Regression_V2(
        seq_length, val_start, val_end,
        data_path=data_path,
        x_feature=x_feature,
        y_feature=y_feature,
        x_scaler=train_dataset.x_scaler,
        y_scaler=train_dataset.y_scaler
    )
    test_dataset = MTX_Dataset_Regression_V2(
        seq_length, test_start, test_end,
        data_path=data_path,
        x_feature=x_feature,
        y_feature=y_feature,
        x_scaler=train_dataset.x_scaler,
        y_scaler=train_dataset.y_scaler
    )
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    #sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


def create_dataloader_no_val(seq_length, config_path, data_path):
    """
    創建回歸任務的 DataLoader，不使用 validation set
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_start = config['dataset']['train']['start_date']
    train_end = config['dataset']['train']['end_date']
    test_start = config['dataset']['test']['start_date']
    test_end = config['dataset']['test']['end_date']
    batch_size = config['dataloader'].get('batch_size', 32)
    features = config['model'].get('features', ['close_denoised'])
    print(f"train start date : {train_start}, train end date : {train_end}")
    print(f"test start date : {test_start}, test end date : {test_end}")
    test_features = config['model'].get('test_features', ['open', 'close', 'high', 'low', 'K', 'D', 'MA20', 'MFI'])
    print(f"\nCreating Train Dataset...")
    train_dataset = MTX_Dataset_Regression_MultiFeature(
        seq_length, train_start, train_end,
        data_path=data_path,
        features=features,
        label='close_denoised',
        trend_col='trend_raw',
        open_col='open_denoised',
        scaler=None
    )
    print(f"\nCreating Test Dataset...")

    test_dataset = MTX_Dataset_Regression_MultiFeature(
        seq_length, test_start, test_end,
        data_path=data_path,
        features=test_features,
        label='close',
        trend_col='trend_raw',
        open_col='open',
        scaler=train_dataset.scaler
    )
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader, train_dataset.feature_count


def train_model_regression(model, train_loader, val_loader, name, lr=0.001):
    """
    回歸任務的訓練函數
    
    Args:
        model: xLSTM 模型
        train_loader: 訓練 DataLoader
        test_loader: 測試 DataLoader
        name: 模型名稱
        lr: 學習率（可通過參數傳入）
    """
    # 超參數
    num_epochs = 200
    patience = 30
    best_val_loss = float('inf')
    trigger_times = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    model.to(device)
    
    # 使用 MSE Loss（回歸任務）
    criterion = nn.MSELoss()
    
    # 優化器 - AdamW（比 Adam 的 weight decay 更強，調低一點）
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    # 學習率調度
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)
    
    print(f'\nTraining Configuration (Regression):')
    print(f'  Task: Price Prediction (Continuous)')
    print(f'  Loss: MSELoss')
    print(f'  Direction Reference: 當天的 open (標準化後)')
    print(f'  Optimizer: Adam (weight decay: 0.0001)')
    print(f'  Initial LR: {lr}')
    print(f'  Max Epochs: {num_epochs}')
    print(f'  Patience: {patience}')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            epoch_loss += loss.item()
            
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        
        avg_train_loss = epoch_loss / len(train_loader)
        # Validation
        model.eval()

        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred = model(batch_x)
                
                # 使用當天的 open（標準化後）作為 base_price 來計算方向
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        
        # 計算 MAE（標準化和百分比）
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 計算方向準確率（Direction Accuracy）
        # 使用反標準化後的 pred 和當天的 open 計算方向
        # pred_direction: 根據 pred - 當天的open 判斷方向
        # trend: 實際方向（0=下跌, 1=持平, 2=上漲 或 0=下跌, 1=上漲）
        
        # 反標準化 pred（從標準化值回到絕對價格）

        preds_denorm = val_loader.dataset.denormalize_y(all_preds).flatten()
        labels_denorm = val_loader.dataset.denormalize_y(all_labels).flatten()
        mae = np.mean(np.abs(preds_denorm - labels_denorm))
        # 計算預測方向：pred - 當天的open

        
        # 學習率調度
        scheduler.step(avg_val_loss)
        
        # 打印進度
        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_val_loss < best_val_loss:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  #f'Train MAE: {train_mae_price:.2f} pts, '
                  f'Test Loss: {avg_val_loss:.6f}, '
                  f'Test MAE: {mae:.2f} pts, '
                  f'LR: {optimiser.param_groups[0]["lr"]:.6f}', flush=True)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{name}.pth')
            print(f'>>> New best model saved! Val Loss: {avg_val_loss:.6f}', flush=True)
            trigger_times = 0
        else:
            trigger_times += 1
            
        # Early stopping: 如果 validation 沒有改善 或 overfitting 太嚴重
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}!', flush=True)
                print(f'Best Val Loss: {best_val_loss:.6f}', flush=True)
                break
    
    print("Training complete!", flush=True)
    print(f'Best Val Loss: {best_val_loss:.6f}', flush=True)
    
    return best_val_loss


def train_model_regression_no_val(model, train_loader, test_loader, name, lr=0.001):
    """
    回歸任務的訓練函數（不使用 validation set，固定 30 個 epoch）
    每個 epoch 結束後用 test_loader 評估性能（不參與訓練）
    
    Args:
        model: xLSTM 模型
        train_loader: 訓練 DataLoader
        test_loader: 測試 DataLoader（僅用於評估，不參與訓練）
        name: 模型名稱
        lr: 學習率（可通過參數傳入）
    """
    # 超參數
    num_epochs = 40  # 固定 30 個 epoch
    best_test_loss = float('inf')
    best_test_acc = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    model.to(device)
    
    # 使用 MSE Loss（回歸任務）
    criterion = DirectionalMSELoss(direction_weight=2.0)
    
    # 優化器 - AdamW（比 Adam 的 weight decay 更強，調低一點）
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    # 學習率調度（基於 train loss）
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)
    
    print(f'\nTraining Configuration (Regression - No Validation):')
    print(f'  Task: Price Prediction (Continuous)')
    print(f'  Loss: DirectionalMSELoss (direction_weight={criterion.direction_weight})')
    print(f'  Direction Reference: 當天的 open (標準化後)')
    print(f'  Optimizer: Adam')
    print(f'  Initial LR: {lr}')
    print(f'  Fixed Epochs: {num_epochs}')
    print(f'  Test Evaluation: After each epoch (not used for training)')
    
    for epoch in range(num_epochs):
        # ========== Training Phase ==========
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y, _, _, batch_open_norm in train_loader:  # trend, open_raw, open_normalized
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_open_norm = batch_open_norm.to(device).unsqueeze(1)  # (batch_size, 1)
            
            pred = model(batch_x)
            # 使用當天的 open（標準化後）作為 base_price 來計算方向
            loss = criterion(pred, batch_y, batch_open_norm)
            epoch_loss += loss.item()
            
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # ========== Test Evaluation Phase (不參與訓練) ==========
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        all_opens = []  # 存儲當天的 open（原始值）
        all_trends = []
        
        with torch.no_grad():
            for idx, (batch_x, batch_y, trend, open_raw, open_normalized) in enumerate(test_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                trend = trend.to(device)
                open_normalized = open_normalized.to(device).unsqueeze(1)  # (batch_size, 1)

                pred = model(batch_x)
                
                # 使用當天的 open（標準化後）作為 base_price 來計算方向
                loss = criterion(pred, batch_y, open_normalized)
                test_loss += loss.item()
                
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                all_opens.append(open_raw)  # open_raw 已經是 numpy array
                all_trends.append(trend.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        # 計算 MAE 和方向準確率
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_opens = np.concatenate(all_opens)  # 轉換為 numpy array（確保正確展平）
        all_trends = np.concatenate(all_trends)
        
        # 反標準化 pred（從標準化值回到絕對價格）
        preds_denorm = test_loader.dataset.denormalize(all_preds).flatten()
        labels_denorm = test_loader.dataset.denormalize(all_labels).flatten()
        mae = np.mean(np.abs(preds_denorm - labels_denorm))
        
        # 計算預測方向：pred - 當天的open
        pred_diff = preds_denorm - all_opens  # pred - 當天的open
        actual_trend = all_trends.flatten()

        # 二分類：0=下跌, 1=上漲
        # pred_direction: 1=上漲 (pred > open), 0=下跌 (pred <= open)
        pred_direction = (pred_diff > 0).astype(int)
        # actual_direction: 0=下跌, 1=上漲
        actual_direction = actual_trend.astype(int)
    
        # 計算方向準確率
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # 學習率調度（基於 train loss）
        scheduler.step(avg_train_loss)
        
        # 打印進度
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, '
              f'Test MAE: {mae:.2f} pts, '
              f'Dir Acc: {direction_accuracy:.2f}%, '
              f'LR: {optimiser.param_groups[0]["lr"]:.6f}', flush=True)
        
        # 保存最佳模型（基於 test 性能）
        if direction_accuracy > best_test_acc or (direction_accuracy == best_test_acc and avg_test_loss < best_test_loss):
            best_test_acc = direction_accuracy
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), f'{name}.pth')
            print(f'>>> New best model saved! Test Loss: {avg_test_loss:.6f} Direction Accuracy: {direction_accuracy:.2f}%', flush=True)
    
    print("Training complete!", flush=True)
    print(f'Best Test Loss: {best_test_loss:.6f} Direction Accuracy: {best_test_acc:.2f}%', flush=True)


def finetune_model_regression(model, finetune_loader, test_loader, name, num_epochs=10, lr=0.0005):
    """
    對已訓練好的模型進行微調（Fine-tuning）
    使用 finetune_loader（train+val 數據）進行訓練
    每個 epoch 用 test_loader 做評估（僅用於監控，不參與訓練）
    保存最佳模型基於 finetune_loader 的性能
    
    Args:
        model: 已訓練好的 xLSTM 模型
        finetune_loader: 微調 DataLoader（train+val 數據）
        test_loader: 測試 DataLoader（僅用於評估，不參與訓練）
        name: 模型名稱（保存時會加上 _finetuned 後綴）
        num_epochs: 微調的 epoch 數（默認 10）
        lr: 學習率（默認 0.0001，通常比初始訓練更小）
    """
    best_finetune_loss = float('inf')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    model.to(device)
    
    # 使用 MSE Loss（回歸任務）
    criterion = DirectionalMSELoss(direction_weight=2.0)
    
    # 優化器 - 使用較小的學習率進行微調
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    # 學習率調度
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)
    
    print(f'\nFine-tuning Configuration:')
    print(f'  Task: Price Prediction (Continuous) - Fine-tuning')
    print(f'  Loss: DirectionalMSELoss (direction_weight={criterion.direction_weight})')
    print(f'  Direction Reference: 當天的 open (標準化後)')
    print(f'  Optimizer: Adam')
    print(f'  Initial LR: {lr}')
    print(f'  Epochs: {num_epochs}')
    print(f'  Finetune Data: train+val')
    print(f'  Test Evaluation: After each epoch (not used for training)')
    print(f'  Best Model Selection: Based on finetune loss')
    
    for epoch in range(num_epochs):
        # ========== Fine-tuning Phase ==========
        model.train()
        finetune_loss = 0
        
        for batch_x, batch_y, _, _, batch_open_norm in finetune_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_open_norm = batch_open_norm.to(device).unsqueeze(1)  # (batch_size, 1)
            
            pred = model(batch_x)
            # 使用當天的 open（標準化後）作為 base_price 來計算方向
            loss = criterion(pred, batch_y, batch_open_norm)
            finetune_loss += loss.item()
            
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        
        avg_finetune_loss = finetune_loss / len(finetune_loader)
        
        # ========== Test Evaluation Phase (僅用於監控，不參與訓練) ==========
        model.eval()
        test_loss = 0
        test_all_preds = []
        test_all_labels = []
        test_all_opens = []
        test_all_trends = []
        
        with torch.no_grad():
            for batch_x, batch_y, trend, open_raw, open_normalized in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                trend = trend.to(device)
                open_normalized = open_normalized.to(device).unsqueeze(1)
                
                pred = model(batch_x)
                loss = criterion(pred, batch_y, open_normalized)
                test_loss += loss.item()
                
                test_all_preds.append(pred.cpu().numpy())
                test_all_labels.append(batch_y.cpu().numpy())
                test_all_opens.append(open_raw)
                test_all_trends.append(trend.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        # 計算 test 的 MAE 和方向準確率
        test_all_preds = np.concatenate(test_all_preds)
        test_all_labels = np.concatenate(test_all_labels)
        test_all_opens = np.concatenate(test_all_opens)
        test_all_trends = np.concatenate(test_all_trends)
        
        # 反標準化
        test_preds_denorm = test_loader.dataset.denormalize(test_all_preds).flatten()
        test_labels_denorm = test_loader.dataset.denormalize(test_all_labels).flatten()
        test_mae = np.mean(np.abs(test_preds_denorm - test_labels_denorm))
        
        # 計算方向準確率
        test_pred_diff = test_preds_denorm - test_all_opens
        test_actual_trend = test_all_trends.flatten()
        test_pred_direction = (test_pred_diff > 0).astype(int)
        test_actual_direction = test_actual_trend.astype(int)
        test_direction_accuracy = np.mean(test_pred_direction == test_actual_direction) * 100
        
        # 學習率調度（基於 finetune loss）
        scheduler.step(avg_finetune_loss)
        
        # 打印進度
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Finetune Loss: {avg_finetune_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, '
              f'Test MAE: {test_mae:.2f} pts, '
              f'Test Dir Acc: {test_direction_accuracy:.2f}%, '
              f'LR: {optimiser.param_groups[0]["lr"]:.6f}', flush=True)
        
        # 保存最佳模型（基於 finetune loss）
        if avg_finetune_loss < best_finetune_loss:
            best_finetune_loss = avg_finetune_loss
            finetuned_name = f'{name}_finetuned.pth'
            torch.save(model.state_dict(), finetuned_name)
            print(f'>>> New best finetuned model saved! Finetune Loss: {avg_finetune_loss:.6f}, '
                  f'Test Dir Acc: {test_direction_accuracy:.2f}%', flush=True)
    
    print("Fine-tuning complete!", flush=True)
    print(f'Best Finetune Loss: {best_finetune_loss:.6f}', flush=True)
    print(f'Final model saved as: {name}_finetuned.pth', flush=True)

def sequential_test(model, model_path, config_path, data_path, seq_length, x_scaler, y_scaler):
    """
    逐步測試函數：對 close 做完整去噪 -> 切 seq_length -> normalize -> predict
    
    流程：
    1. 讀取原始數據（包含 close）
    2. 對整個 close 序列進行完整去噪
    3. 按照 seq_length 切分序列
    4. 使用訓練時的 scaler 進行標準化
    5. 使用模型進行預測
    
    Args:
        model: xLSTM 模型
        model_path: 模型權重路徑
        config_path: 配置文件路徑（用於獲取測試日期範圍）
        data_path: 原始數據路徑（包含 close）
        seq_length: 序列長度
        x_scaler: 訓練時的 MinMaxScaler（從 train_dataset.scaler 獲取）
        y_scaler: 訓練時的 MinMaxScaler（從 train_dataset.scaler 獲取）
    
    Returns:
        results: dict，包含預測結果、真實值、日期等信息
    """
    print("\n" + "="*80)
    print("Sequential Test: Denoise -> Slice -> Normalize -> Predict")
    print("="*80)
    
    # 讀取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    test_start = config['dataset']['test']['start_date']
    test_end = config['dataset']['test']['end_date']
    
    print(f"\nTest period: {test_start} to {test_end}")
    
    # Step 1: 讀取原始數據並找到測試期間的索引
    print("\nStep 1: Reading raw data and finding test period indices...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 找到測試期間的索引
    start_idx = df.index[df['date'] >= test_start][0] if len(df.index[df['date'] >= test_start]) > 0 else None
    end_idx = df.index[df['date'] <= test_end][-1] if len(df.index[df['date'] <= test_end]) > 0 else None
    
    if start_idx is None or end_idx is None:
        raise ValueError(f"Cannot find data for test period: {test_start} to {test_end}")
    
    print(f"  Test start index: {start_idx}, Test end index: {end_idx}")
    print(f"  Test period: {df['date'].iloc[start_idx]} to {df['date'].iloc[end_idx]}")
    print(f"  Number of test samples: {end_idx - start_idx + 1}")
    
    # 讀取完整數據
    close_raw_full = df['close'].values
    test_dates = df['date'].iloc[start_idx:end_idx+1].values
    test_close_raw = df['close'].iloc[start_idx:end_idx+1].values
    
    # Step 2: 對每個測試索引進行去噪並創建序列
    print(f"\nStep 2: Denoising [0: i] and taking last {seq_length} values for each test index...")
    all_preds = []
    all_labels = []  # 用於存儲真實值（原始 close，因為我們預測的是去噪後的）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(f'{model_path}.pth'))
    model.eval()
    with torch.no_grad():
        for i in range(start_idx, end_idx + 1):
            # Step 1: 對 [0: i] 進行去噪
            close_raw_until_i = close_raw_full[0:i]
            close_denoised = wavelet_denoising(close_raw_until_i)
            
            # Step 2: 取最後 seq_length 個作為序列
            seq = close_denoised[-seq_length:]
            seq_normalized = x_scaler.transform(seq.reshape(-1, 1))
            seq_tensor = torch.tensor(seq_normalized, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_tensor)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(close_raw_full[i])  # 真實值是原始 close[i]
            
            if (i - start_idx + 1) % 50 == 0:
                print(f"    Processed {i - start_idx + 1}/{end_idx - start_idx + 1} samples...", flush=True)
    # Step 5: 預測
    print("\nStep 5: Making predictions...")

    all_preds = np.concatenate(all_preds)
    preds_denorm = y_scaler.inverse_transform(all_preds).flatten()
    labels_denorm = all_labels
    # 計算指標
    mae = np.mean(np.abs(preds_denorm - labels_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - labels_denorm) ** 2))
    
    print("\n" + "="*80)
    print("Sequential Test Results:")
    print("="*80)
    print(f"  Number of samples: {len(all_preds)}")
    print(f"  MAE: {mae:.2f} pts")
    print(f"  RMSE: {rmse:.2f} pts")
    print("="*80)
    
    # 保存結果到 CSV
    results_df = pd.DataFrame({
        'date': test_dates,
        'pred_denorm': preds_denorm,
        'label_denorm': labels_denorm,
        'close_raw': test_close_raw,
        'error': preds_denorm - labels_denorm,
        'abs_error': np.abs(preds_denorm - labels_denorm)
    })
    
    # 生成輸出文件名（基於模型名稱）
    output_csv = f'{model_path}_sequential_test_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # 返回結果
    results = {
        'dates': test_dates,
        'predictions': preds_denorm,
        'labels': labels_denorm,
        'close_raw': test_close_raw,
        'mae': mae,
        'rmse': rmse,
        'num_samples': len(all_preds),
        'csv_path': output_csv
    }
    
    return results


def evaluate_model(model , test_loader, name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(f'{name}.pth'))
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            test_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    preds_denorm = test_loader.dataset.denormalize_y(all_preds).flatten()
    labels_denorm = test_loader.dataset.denormalize_y(all_labels).flatten()
    mae = np.mean(np.abs(preds_denorm - labels_denorm))
    
    print(f'Test Loss: {avg_test_loss:.6f}, Test MAE: {mae:.2f} pts', flush=True)
    
    return avg_test_loss, mae
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='模型名稱')
    parser.add_argument('--seq', default=150, type=int, help='序列長度')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--seed', default=42, type=int, help='隨機種子')
    parser.add_argument('--data_path', type=str, 
                        default=None,
                        help='Train 數據路徑（去噪），若為 None 則自動推斷')
    parser.add_argument('--lr', type=float, 
                        default=0.001,
                        help='學習率')
    args = parser.parse_args()
    
    print("="*80)
    print("xLSTM REGRESSION TRAINING - PAPER STYLE")
    print("="*80)

    set_random_seed(args.seed)
    
    print(f"\nData configuration:")
    print(f"  Train: {args.data_path} (denoised)")
    train_loader, val_loader, test_loader = create_dataloader_regression(args.seq, args.config, args.data_path)
    xlstm_stack, input_projection, output_projection = create_xlstm_model(args.seq)
    
    model = xLSTMClassifier(input_projection, xlstm_stack, output_projection)   
    train_model_regression(model, train_loader, val_loader, args.name, lr=args.lr)
    evaluate_model(model, test_loader, args.name)
    #sequential_test(model, args.name, args.config, args.data_path, args.seq, train_loader.dataset.x_scaler, train_loader.dataset.y_scaler)
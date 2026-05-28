import os
os.environ["XLSTM_EXTRA_INCLUDE_PATHS"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
print(os.environ.get("XLSTM_EXTRA_INCLUDE_PATHS"))
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from MTX_DataSet import *
from xLSTM_TS import *
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from utils import *
import yaml
from argparse import ArgumentParser

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def set_random_seed(seed):
    # 設定 Python 原生的隨機種子
    random.seed(seed)
    
    # 設定 NumPy 的隨機種子
    np.random.seed(seed)
    
    # 設定 PyTorch 的隨機種子
    torch.manual_seed(seed)
    
    # 如果有使用 CUDA，設定 CUDA 隨機數種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多個 GPU，設置所有 GPU 的種子
    
    # 設定 cudnn 的隨機性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_binary(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            predicted = (torch.sigmoid(pred) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    if total == 0:
        return 0.0, 0.0, np.array([[0, 0], [0, 0]])
    acc = correct / total * 100
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, f1, conf_matrix


def build_train_loader(dataset, batch_size):
    effective_labels = dataset.labels.squeeze().cpu().long()[dataset.seq_length:]
    class_counts = torch.bincount(effective_labels, minlength=2).float().clamp(min=1.0)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[effective_labels].double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 打印详细的平衡信息
    total_samples = len(effective_labels)
    down_count = class_counts[0].item()
    up_count = class_counts[1].item()
    down_pct = (down_count / total_samples) * 100
    up_pct = (up_count / total_samples) * 100
    
    print(f'\n{"="*60}')
    print(f'Train Data Balance Analysis')
    print(f'{"="*60}')
    print(f'Total samples (after seq shift): {total_samples}')
    print(f'Down (0): {int(down_count):5d} ({down_pct:5.2f}%)')
    print(f'Up   (1): {int(up_count):5d} ({up_pct:5.2f}%)')
    print(f'Imbalance ratio: {abs(down_count - up_count) / total_samples * 100:.2f}%')
    print(f'\nClass weights:')
    print(f'  Down (0): {class_weights[0].item():.4f}')
    print(f'  Up   (1): {class_weights[1].item():.4f}')
    print(f'{"="*60}\n')
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def create_dataloader(seq_length, config_path, data_path):

    config = load_config(config_path)
    batch_size = config["dataloader"]["batch_size"]
    train_cfg = config["dataset"]["train"]
    val_cfg = config["dataset"]["val"]
    test_cfg = config["dataset"]["test"]

    train_features = config.get("model", {}).get("features", ["close_denoised"])
    #train_features = ['close_denoised']
    train_dataset = MTX_Dataset(seq_length, train_cfg["start_date"], train_cfg["end_date"], data_path=data_path, features=train_features, label='trend_raw')
    val_dataset = MTX_Dataset(seq_length, val_cfg["start_date"], val_cfg["end_date"], scaler_norm=train_dataset.scaler_norm, data_path=data_path, features=train_features, label='trend_raw')
    test_dataset = MTX_Dataset(seq_length, test_cfg["start_date"], test_cfg["end_date"], scaler_norm=train_dataset.scaler_norm, data_path=data_path, features=train_features, label='trend_raw')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'train start date : {train_cfg["start_date"]}, train end date : {train_cfg["end_date"]}')
    print(f'val start date : {val_cfg["start_date"]}, val end date : {val_cfg["end_date"]}')
    print(f'test start date : {test_cfg["start_date"]}, test end date : {test_cfg["end_date"]}')
    print(f'Features used: {train_dataset.features}')
    print(f'Feature count: {train_dataset.feature_count}')
    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Val dataset size: {len(val_loader.dataset)}')
    print(f'Test dataset size: {len(test_loader.dataset)}')
    print(f'Input data shape: {train_dataset.data.shape}')
    print(f'Normalized data shape: {train_dataset.normalized_data.shape}')
    return train_loader, val_loader, test_loader


def train_model(xLSTM_classifier, train_loader, val_loader, name, lr, num_epochs, patience, scheduler_patience, model_dir):
    best_val_loss = float('inf')
    best_val_acc = 0
    trigger_times = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check if GPU is available and set the device
    print(f'lr: {lr}')
    print(f'epochs: {num_epochs}')
    print(f'Using device: {device}')
    print(f'early stopping patience: {patience}')
    # Move the model to the device (GPU or CPU)
    xLSTM_classifier.to(device)    
    
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(xLSTM_classifier.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=scheduler_patience)
    
    # Trainin model
    for epoch in range(num_epochs):
        xLSTM_classifier.train()
        
        correct_train = 0
        total_train = 0      
        all_train_preds = []
        all_train_labels = []
        epoch_loss = 0
        # logits_list = []          
        for batch_x, batch_y in train_loader:
            # Move data to the selected device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = xLSTM_classifier(batch_x) 
            loss = criterion(pred, batch_y)
            epoch_loss += loss.item()
            # # log logits
            # logits_list.append(pred.cpu())

            # Calculate accuracy for training
            predicted = (torch.sigmoid(pred) > 0.5).float()  # Threshold at 0.5 for binary classification
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

            optimiser.zero_grad()
            loss.backward()            
            torch.nn.utils.clip_grad_norm_(xLSTM_classifier.parameters(), max_norm=1.0)  # Apply gradient clipping to prevent the exploding gradient problem
            optimiser.step()

        epoch_loss /= len(train_loader)
        # plot train logit and simoid distribution
        # all_logits = torch.cat(logits_list, dim=0)
        # plot_logits_distribution(all_logits, logits_type="raw") 
        # plot_logits_distribution(all_logits, logits_type="sigmoid")
        
        # Calculate training accuracy
        train_acc = correct_train / total_train * 100

        # Calculate F1 score for training
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        
        # Calculate confusion matrix
        train_conf_matrix = confusion_matrix(all_train_labels, all_train_preds)        
        
        # validate model
        xLSTM_classifier.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        
                
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move data to the selected device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred = xLSTM_classifier(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss

                # Calculate accuracy for validation
                predicted = (torch.sigmoid(pred) > 0.5).float()  # Threshold at 0.5 for binary classification
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
                
                all_val_preds.extend(predicted.cpu().numpy())  # Collecting predictions for F1 score
                all_val_labels.extend(batch_y.cpu().numpy())  # Collecting true labels


        # Calculate validation accuracy
        val_acc = correct_val / total_val * 100

        # Calculate F1 score for validation
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_val_labels, all_val_preds)

               
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print training and validation progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.8f}, Validation Loss: {val_loss:.8f}, '
              f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%, '
              f'Train F1 Score: {train_f1:.4f}, Validation F1 Score: {val_f1:.4f}')
        
        print(f'train Confusion Matrix (Epoch {epoch + 1}):\n{train_conf_matrix}')   
        print(f'Val Confusion Matrix (Epoch {epoch + 1}):\n{conf_matrix}')
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(xLSTM_classifier.state_dict(), f'{model_dir}/{name}.pth')
            print(f'New best model saved! val loss: {val_loss:.8f}, val acc: {val_acc:.2f}%')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break            
            
    print("Training complete!")


if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='The name of the saved file (without extension).')
    parser.add_argument('--model_dir', type=str, required=True, help='The directory to save the model.')
    parser.add_argument('--seq', default=100, type=int, help='The sequence length the model looks back on')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    parser.add_argument('--data_path', type=str, default="TradeSelector_processed_data/incremental_denoised_data.csv",
                        help='Path to data file')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for LR scheduler')
    args = parser.parse_args()
    
    
    set_random_seed(args.seed)
    train_loader, val_loader, test_loader = create_dataloader(
        args.seq,
        args.config,
        args.data_path,
    )
    
    # 根據特徵數量創建模型
    input_size = train_loader.dataset.feature_count
    print(f'Creating xLSTM model with input size: {input_size}')
    
    xlstm_stack, input_projection, output_projection = create_xlstm_model(args.seq, input_size, embedding_dim=64)
    model = xLSTMClassifier(input_projection, xlstm_stack, output_projection)

    train_model(
        model,
        train_loader,
        val_loader,
        args.name,
        args.lr,
        args.epochs,
        args.patience,
        args.scheduler_patience,
        args.model_dir
    )
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 評估驗證後的最佳模型
    stage1_path = f'{args.model_dir}/{args.name}.pth'
    model.load_state_dict(torch.load(stage1_path))
    model.to(device)

    test_acc, test_f1, test_conf_matrix = evaluate_binary(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}')
    print(f'Test Confusion Matrix:\n{test_conf_matrix}')

    val_acc, val_f1, val_conf_matrix = evaluate_binary(model, val_loader, device)
    print(f'Validation Accuracy (reference): {val_acc:.2f}%')
    print(f'Validation F1 Score (reference): {val_f1:.4f}')
    print(f'Validation Confusion Matrix (reference):\n{val_conf_matrix}')
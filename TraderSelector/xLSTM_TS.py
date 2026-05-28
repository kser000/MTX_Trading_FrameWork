# src/ml/models/xlstm_ts/xlstm_ts_model.py

# -------------------------------------------------------------------------------------------
# New proposed model: xLSTM-TS, a time series-specific implementation
# 
# References:
# 
# - Paper (2024): https://doi.org/10.48550/arXiv.2405.04517
# - Official code: https://github.com/NX-AI/xlstm
# - Parameters for time series: https://github.com/smvorwerk/xlstm-cuda
# -------------------------------------------------------------------------------------------
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from torchinfo import summary

# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------

def create_xlstm_model(seq_length, input_size=1, embedding_dim=64):
    # Define your input size, hidden size, and other relevant parameters
    output_size = 1  # Number of output features (predicting the next value)

    # Define the xLSTM configuration    
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=2, num_heads=2  # Reduced parameters to save memory
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=2,  # Reduced number of heads to save memory
                conv1d_kernel_size=2,  # Reduced kernel size to save memory
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),  # Reduced projection factor to save memory
        ),
        context_length=seq_length,
        num_blocks=4,  # Reduced number of blocks to save memory
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )

    # Instantiate the xLSTM stack
    xlstm_stack = xLSTMBlockStack(cfg)

    # Add  a linear layer to project input data to the required embedding dimension
    input_projection = nn.Linear(input_size, embedding_dim)

    # Add a final linear layer to project the xLSTM output to the desired output size
    output_projection = nn.Linear(embedding_dim, output_size)

    return xlstm_stack, input_projection, output_projection

def create_xlstm_model_small(seq_length, input_size=1):
    """
    創建小型 xLSTM 模型，適合小數據集（1550-2000 樣本）
    
    平衡策略：適度容量 + 適當正則化
    - embedding_dim: 40 (32太小導致欠擬合，64太大容易過擬合)
    - num_blocks: 3 (減少深度，降低參數)
    - 其他參數適度縮減
    """
    embedding_dim = 40  # 32->40，適中容量
    output_size = 1

    # 調整的 xLSTM 配置
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=3,  # 2->3，適度增加（2太小可能限制學習）
                qkv_proj_blocksize=1,  # 保持 1
                num_heads=2  # 保持 2
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=2,
                conv1d_kernel_size=2,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.05, act_fn="gelu"),  # 1.1->1.05，適度縮減
        ),
        context_length=seq_length,
        num_blocks=3,  # 4->3，減少深度
        embedding_dim=embedding_dim,
        slstm_at=[1],  # 只在第1個block用sLSTM
    )

    # 實例化 xLSTM stack
    xlstm_stack = xLSTMBlockStack(cfg)

    # 輸入投影層
    input_projection = nn.Linear(input_size, embedding_dim)

    # 輸出投影層
    output_projection = nn.Linear(embedding_dim, output_size)

    return xlstm_stack, input_projection, output_projection

def create_xlstm_model_medium(seq_length, input_size=1):
    """
    創建中型 xLSTM 模型，介於原始和小型之間
    
    目標參數: ~80K (從 125K 縮減 35%)
    """
    embedding_dim = 32  # 64 -> 48 (縮減 25%)
    output_size = 1

    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,  # 4 -> 3
                qkv_proj_blocksize=2,  # 2 -> 1
                num_heads=2  # 保持 2
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=2,  # 保持 2
                conv1d_kernel_size=2,  # 保持 2
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),  # 1.1 -> 1.05
        ),
        context_length=seq_length,
        num_blocks=3,  # 4 -> 3 (縮減 25%)
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )

    xlstm_stack = xLSTMBlockStack(cfg)
    input_projection = nn.Linear(input_size, embedding_dim)
    output_projection = nn.Linear(embedding_dim, output_size)

    return xlstm_stack, input_projection, output_projection
    
class xLSTMClassifier(nn.Module):
    def __init__(self, input_projection, xlstm_stack, output_projection, dropout=0):
        super().__init__()
        self.input_projection = input_projection
        self.xlstm_stack = xlstm_stack
        self.dropout = nn.Dropout(dropout)  # 添加 dropout 緩解 overfitting
        self.output_projection = output_projection

    def forward(self, x):
        x = self.input_projection(x)
        x = self.xlstm_stack(x)
        x = self.dropout(x[:, -1, :])  # 應用 dropout
        x = self.output_projection(x)

        return x
    
def create_xlstm_classifier(input_dim=1, seq_length=100, num_classes=1, dropout=0):
    """
    創建 xLSTM 分類器
    
    Args:
        input_dim: 輸入特徵數量
        seq_length: 序列長度
        num_classes: 輸出類別數 (1 for binary, 3 for 3-class)
        dropout: Dropout 比率，預設 0.3（較高的 dropout 減少 overfitting）
    
    Returns:
        xLSTMClassifier model
    """
    embedding_dim = 64
    
    # Define the xLSTM configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=2, num_heads=2
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=2,
                conv1d_kernel_size=2,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),
        ),
        context_length=seq_length,
        num_blocks=2,
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )
    
    xlstm_stack = xLSTMBlockStack(cfg)
    input_projection = nn.Linear(input_dim, embedding_dim)
    output_projection = nn.Linear(embedding_dim, num_classes)
    
    return xLSTMClassifier(input_projection, xlstm_stack, output_projection, dropout=dropout)

def plot_architecture_xlstm(input_size=1):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(100, input_size)

    model = xLSTMClassifier(input_projection, xlstm_stack, output_projection).cuda()

    batch_size = 16
    real_input_dimensions = (batch_size, 100, input_size)

    # Generate the summary with actual input dimensions
    summary(model, input_size=real_input_dimensions)
import torch
import torch.nn as nn
import os
from layers.RevIN import RevIN
from layers.PatchTST_backbone import TSTiEncoder


class PatchTST_Classifier(nn.Module):
    """
    PatchTST 二分类模型
    直接构建所需组件（RevIN + Patching + Encoder + Classifier Head）
    输出 logits（不使用 Sigmoid，因为使用 BCEWithLogitsLoss）
    
    支持两种模式：
    - separate=False: 输出整日 trend (d_model -> 1)，使用单个 classifier
    - separate=True: 输出夜日盘分开的 trend (d_model -> 2)，使用两个独立的 classifier
        - classifier_night: 处理夜盘 trend (d_model -> 1)
        - classifier_day: 处理日盘 trend (d_model -> 1)
    """
    def __init__(self, args, n_features):
        super(PatchTST_Classifier, self).__init__()
        
        # 保存 args 以便后续加载预训练模型
        self.args = args
        
        # 参数设置
        self.revin = getattr(args, 'revin', True)
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.padding_patch = getattr(args, 'padding_patch', 'end')
        self.separate = getattr(args, 'separate', False)  # 是否使用夜日盘分开的 trend
        
        # 计算 patch_num
        patch_num = int((args.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch == 'end':
            patch_num += 1
        
        # RevIN normalization
        if self.revin:
            affine = getattr(args, 'affine', False)
            subtract_last = getattr(args, 'subtract_last', False)
            self.revin_layer = RevIN(n_features, affine=affine, subtract_last=subtract_last)
        
        # Patching padding layer
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        
        # Encoder (backbone)
        self.backbone = TSTiEncoder(
            c_in=n_features,
            patch_num=patch_num,
            patch_len=self.patch_len,
            max_seq_len=1024,
            n_layers=args.e_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_k=None,
            d_v=None,
            d_ff=args.d_ff,
            norm='BatchNorm',
            attn_dropout=0.,
            dropout=args.dropout,
            act='gelu',
            key_padding_mask='auto',
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe='zeros',
            learn_pe=True,
            verbose=False
        )
        
        # Classifier Head
        # backbone 输出: [bs x nvars x d_model x patch_num]
        # Step 1: time pooling -> [bs x nvars x d_model]
        # Step 2: feature mean pooling -> [bs x d_model]
        # Step 3: classification -> [bs x 1] 或 [bs x 2]
        self.d_model = args.d_model
        self.time_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 整日 trend 分类器 (d_model -> 1)
        self.classifier = nn.Sequential(
            nn.Dropout(args.head_dropout),
            nn.Linear(self.d_model, 1)
        )
        
        # 夜日盘分开的 trend 分类器
        if self.separate:
            # 方案：保留 nvars 维度，使用注意力机制学习不同特征的重要性
            # 不使用简单的 mean pooling，而是让模型学习哪些特征对夜盘/日盘更重要
            
            # 夜盘注意力机制：学习哪些特征对夜盘更重要
            # 输入: [bs, nvars, d_model]，输出注意力权重: [bs, nvars, 1]
            self.night_attention = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1)
            )
            
            # 日盘注意力机制：学习哪些特征对日盘更重要
            # 输入: [bs, nvars, d_model]，输出注意力权重: [bs, nvars, 1]
            self.day_attention = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1)
            )
            
            # 夜盘分类器 (d_model -> 1)
            self.classifier_night = nn.Sequential(
                nn.Dropout(args.head_dropout),
                nn.Linear(self.d_model, 1)
            )
            
            # 日盘分类器 (d_model -> 1)
            self.classifier_day = nn.Sequential(
                nn.Dropout(args.head_dropout),
                nn.Linear(self.d_model, 1)
            )
        else:
            # 非 separate 模式：保持原有设计
            self.classifier_night = None
            self.classifier_day = None
        
    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        x = x.permute(0, 2, 1)  # [batch, n_vars, seq_len]
        
        # RevIN normalization
        if self.revin:
            x = x.permute(0, 2, 1)  # [batch, seq_len, n_vars]
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)  # [batch, n_vars, seq_len]
        
        # Patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs x nvars x patch_num x patch_len]
        x = x.permute(0, 1, 3, 2)  # [bs x nvars x patch_len x patch_num]
        
        # Encoder
        x = self.backbone(x)  # [bs x nvars x d_model x patch_num]
        
        # Time Pooling: [bs x nvars x d_model x patch_num] -> [bs x nvars x d_model]
        bs, nvars, d_model, patch_num = x.shape
        x = x.permute(0, 1, 3, 2)  # [bs x nvars x patch_num x d_model]
        x = x.reshape(bs * nvars, patch_num, d_model)  # [bs*nvars x patch_num x d_model]
        x = x.permute(0, 2, 1)  # [bs*nvars x d_model x patch_num]
        x = self.time_pooling(x)  # [bs*nvars x d_model x 1]
        x = x.squeeze(-1).reshape(bs, nvars, d_model)  # [bs x nvars x d_model]
        
        # Classification: 根据 separate 参数选择分类器
        if self.separate:
            # 夜日盘分开的 trend: 使用注意力机制学习不同特征的重要性
            # x: [bs x nvars x d_model]
            
            # 夜盘：使用注意力机制加权聚合特征
            # 计算每个特征对夜盘的重要性权重
            night_scores = self.night_attention(x)  # [bs x nvars x 1]
            night_weights = torch.softmax(night_scores, dim=1)  # [bs x nvars x 1]，在 nvars 维度归一化
            night_features = torch.sum(x * night_weights, dim=1)  # [bs x d_model]，加权求和
            night_logit = self.classifier_night(night_features)  # [bs x 1]
            
            # 日盘：使用注意力机制加权聚合特征
            # 计算每个特征对日盘的重要性权重
            day_scores = self.day_attention(x)  # [bs x nvars x 1]
            day_weights = torch.softmax(day_scores, dim=1)  # [bs x nvars x 1]，在 nvars 维度归一化
            day_features = torch.sum(x * day_weights, dim=1)  # [bs x d_model]，加权求和
            day_logit = self.classifier_day(day_features)  # [bs x 1]
            
            # 拼接结果
            x = torch.cat([night_logit, day_logit], dim=1)  # [bs x 2]
        else:
            # 整日 trend: 使用简单平均池化
            x = x.mean(dim=1)  # [bs x d_model]
            x = self.classifier(x)  # [bs x 1] (trend logit)
        
        return x  # [batch, 1] 或 [batch, 2]
    
    def load_pretrained_encoder(self, pretrained_path):
        """
        从预训练的 PatchTST 模型直接加载 encoder
        
        Args:
            pretrained_path: 预训练模型 checkpoint 路径
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        
        # 加载预训练模型
        from models import PatchTST
        pretrained_model = PatchTST.Model(self.args).float()
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        pretrained_model.load_state_dict(pretrained_state)
        
        # 提取 encoder (backbone)
        if hasattr(pretrained_model, 'model'):
            pretrained_backbone = pretrained_model.model
        elif hasattr(pretrained_model, 'model_trend'):
            pretrained_backbone = pretrained_model.model_trend
        else:
            raise ValueError("Cannot find encoder in pretrained model")
        
        # 直接替换 encoder 组件
        if self.revin and pretrained_backbone.revin:
            self.revin_layer.load_state_dict(pretrained_backbone.revin_layer.state_dict())
        
        if self.padding_patch == 'end' and pretrained_backbone.padding_patch == 'end':
            self.padding_patch_layer.load_state_dict(pretrained_backbone.padding_patch_layer.state_dict())
        
        self.backbone.load_state_dict(pretrained_backbone.backbone.state_dict())
        
        print(f"Loaded pretrained encoder from: {pretrained_path}")

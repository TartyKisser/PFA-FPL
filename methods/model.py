import torch
import torch.nn as nn
import clip
from methods.template import MLLTemplate
from itertools import combinations
import torch.nn.functional as F
import numpy as np

def create_pairwise_samples(X, K, y_support):
    """
    将图片向量两两拼接，生成样本对
    
    Args:
        X: 图片特征矩阵，形状 (10, 49, 1024)
        K: K矩阵特征，形状 (10, 49, 2048)
        y_support: 标签矩阵，形状 (10, 10) - one-hot编码
    
    Returns:
        K_pairs_forward: 正向拼接的K矩阵对，形状 (45, 49, 2048)
        K_pairs_backward: 反向拼接的K矩阵对，形状 (45, 49, 2048)
        y_pairs: 新标签矩阵（并集），形状 (45, 10)
    """
    
    batch_size = X.shape[0]  # 10
    seq_len = X.shape[1]     # 49
    x_dim = X.shape[2]       # 1024
    k_dim = K.shape[2]       # 2048
    
    # 生成所有可能的配对索引 C(10,2) = 45
    indices = list(combinations(range(batch_size), 2))
    num_pairs = len(indices)  # 45
    
    # 初始化输出矩阵
    X_pairs_forward = torch.zeros(num_pairs, seq_len, 2 * x_dim, device=X.device, dtype=X.dtype)
    X_pairs_backward = torch.zeros(num_pairs, seq_len, 2 * x_dim, device=X.device, dtype=X.dtype)
    K_pairs = torch.zeros(num_pairs, seq_len, 2 * k_dim, device=K.device, dtype=K.dtype)
    y_pairs = torch.zeros(num_pairs, y_support.shape[1], device=y_support.device, dtype=y_support.dtype)
    
    # 生成所有配对
    for idx, (i, j) in enumerate(indices):
        X_pairs_forward[idx] = torch.cat([X[i], X[j]], dim=1)
        X_pairs_backward[idx] = torch.cat([X[j], X[i]], dim=1)
        K_pairs[idx] = torch.cat([K[i], K[j]], dim=1)
        
        # 标签取并集（对于one-hot编码，使用逻辑或）
        y_pairs[idx] = torch.logical_or(y_support[i], y_support[j]).float()
    
    return X_pairs_forward, X_pairs_backward, K_pairs, y_pairs

def compute_X_class(X, H, y):
    # 步骤1: 逐元素相乘
    element_product = torch.einsum('brd,cd->brcd', X, H)  # (batch, num_region, num_class, dim)
    
    # 步骤2: softmax
    softmax_weights = F.softmax(element_product, dim=-1)  # (batch, num_region, num_class, dim)
    
    # 步骤3: 加权特征计算并用y掩码求和
    X_class = torch.einsum('brd,brcd,bc->dc', X, softmax_weights, y)
    
    return X_class

class Model(MLLTemplate):
    def __init__(self, n_way, n_shot, n_query, device='cuda:0', verbose=False, clip_model_name='RN50'):
        super(Model, self).__init__(n_way=n_way, n_shot=n_shot, n_query=n_query,
                                      device=device, verbose=verbose)
        # Load CLIP model with ResNet50
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()  # Set to eval mode
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Get feature dimension from CLIP
        self.feature_dim = self.clip_model.visual.output_dim
        self.mlp_PFA = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True)
        )
        self.MLP = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.n_way)
        )
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.sigma1 = nn.Parameter(torch.tensor(1.0))
        self.sigma2 = nn.Parameter(torch.tensor(1.0))
        self.sigma3 = nn.Parameter(torch.tensor(1.0))
        self.sigma4 = nn.Parameter(torch.tensor(1.0))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)

        self.inference_times = []
    
    def modified_visual_encoder(self, x):
        # Get the visual model (ResNet50)
        visual_model = self.clip_model.visual
        
        # Ensure input is on correct device
        x = x.to(self.device).type(self.clip_model.dtype)
        
        # Forward through ResNet50 stages
        # Initial conv + bn + relu + maxpool
        x = visual_model.relu1(visual_model.bn1(visual_model.conv1(x)))
        x = visual_model.relu2(visual_model.bn2(visual_model.conv2(x)))
        x = visual_model.relu3(visual_model.bn3(visual_model.conv3(x)))
        x = visual_model.avgpool(x)
        
        # ResNet layers
        x = visual_model.layer1(x)
        x = visual_model.layer2(x)
        x = visual_model.layer3(x)
        x = visual_model.layer4(x)
        
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        
        X = visual_model.attnpool.v_proj(x)
        X = visual_model.attnpool.c_proj(X)
        K = visual_model.attnpool.k_proj(x)
        X = X.permute(1, 0, 2)
        K = K.permute(1, 0, 2)

        return X, K
    
    def set_forward(self, x_support, y_support, x_query, selected_values):

        y_support = y_support.float()
        X_support, K = self.modified_visual_encoder(x_support)
        X_support_count = self.clip_model.encode_image(x_support)
        X_query = self.clip_model.encode_image(x_query)
        X_support = X_support.float()
        X_support_count = X_support_count.float()
        X_query = X_query.float()
        
        X_pairs_forward, X_pairs_backward, K_pairs, y_pairs = create_pairwise_samples(X_support, K, y_support)
        X_pairs_forward = self.mlp_PFA(X_pairs_forward)
        X_pairs_backward = self.mlp_PFA(X_pairs_backward)
        K_mean = torch.mean(K_pairs, dim=1, keepdim=True)
        attention_scores = torch.bmm(K_pairs, K_mean.transpose(1, 2))  # (45, 49, 1)
        attention_scores = torch.softmax(attention_scores, dim=1)  # (45, 49, 1)
        
        text_inputs = clip.tokenize(selected_values).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)  # (10, 1024)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        X_class = compute_X_class(X_support, text_features, y_support)
        X_pairs_class = compute_X_class(X_pairs_forward, text_features, y_pairs)

        image_prototype = (X_pairs_class + X_class) / (y_support.sum(dim=0) + y_pairs.sum(dim=0)).unsqueeze(0)
        text_prototype = text_features.transpose(0, 1)  # (1024, 10)
        alpha = torch.exp(self.alpha1) / (torch.exp(self.alpha1) + torch.exp(self.alpha2))
        prototype = alpha * image_prototype + (1 - alpha) * text_prototype  # (1024, 10)
        # prototype = text_prototype.float()
        
        X_squared_norms = torch.sum(X_query ** 2, dim=1, keepdim=True)
        P_squared_norms = torch.sum(prototype ** 2, dim=0, keepdim=True)
        squared_distances = X_squared_norms + P_squared_norms - 2 * torch.matmul(X_query, prototype)
        squared_distances = torch.clamp(squared_distances, min=0.0)
        exp_neg_distances = torch.exp(-torch.sqrt(squared_distances))
        probabilities = exp_neg_distances / torch.sum(exp_neg_distances, dim=1, keepdim=True)
             
        return probabilities
    
    def set_forward_loss(self, x_support, y_support, x_query, y_query, selected_values):
        y_support = y_support.float()
        X_support, K = self.modified_visual_encoder(x_support)
        X_support_count = self.clip_model.encode_image(x_support)
        X_query = self.clip_model.encode_image(x_query)
        X_support = X_support.float()
        X_support_count = X_support_count.float()
        X_query = X_query.float()
        
        X_pairs_forward, X_pairs_backward, K_pairs, y_pairs = create_pairwise_samples(X_support, K, y_support)
        X_pairs_forward = self.mlp_PFA(X_pairs_forward)
        X_pairs_backward = self.mlp_PFA(X_pairs_backward)
        K_mean = torch.mean(K_pairs, dim=1, keepdim=True)
        attention_scores = torch.bmm(K_pairs, K_mean.transpose(1, 2))  # (45, 49, 1)
        attention_scores = torch.softmax(attention_scores, dim=1)  # (45, 49, 1)
        s = torch.sum(X_pairs_forward * attention_scores, dim=1)  # (45, 1024)
        
        text_inputs = clip.tokenize(selected_values).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)  # (10, 1024)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        X_class = compute_X_class(X_support, text_features, y_support)
        X_pairs_class = compute_X_class(X_pairs_forward, text_features, y_pairs)

        image_prototype = (X_pairs_class + X_class) / (y_support.sum(dim=0) + y_pairs.sum(dim=0)).unsqueeze(0)
        text_prototype = text_features.transpose(0, 1)  # (1024, 10)
        alpha = torch.exp(self.alpha1) / (torch.exp(self.alpha1) + torch.exp(self.alpha2))
        prototype = alpha * image_prototype + (1 - alpha) * text_prototype  # (1024, 10)
        # prototype = text_prototype.float()

        ########################Loss_sym########################
        Loss_sym = nn.MSELoss()(X_pairs_forward, X_pairs_backward)
        ########################Loss_sym########################

        ########################Loss_ce########################
        X_ce = torch.cat([X_query, s], dim=0)
        y_ce = torch.cat([y_query, y_pairs], dim=0)

        X_squared_norms = torch.sum(X_ce ** 2, dim=1, keepdim=True)
        P_squared_norms = torch.sum(prototype ** 2, dim=0, keepdim=True)
        squared_distances = X_squared_norms + P_squared_norms - 2 * torch.matmul(X_ce, prototype)
        squared_distances = torch.clamp(squared_distances, min=0.0)
        exp_neg_distances = torch.exp(-torch.sqrt(squared_distances))
        probabilities = exp_neg_distances / torch.sum(exp_neg_distances, dim=1, keepdim=True)
        Loss_ce = nn.BCELoss()(probabilities, y_ce)
        ########################Loss_ce########################

        ########################Loss_count########################
        X_count = torch.cat([X_support_count, X_query], dim=0)
        y_count = torch.cat([y_support, y_query], dim=0)
        count_predictions = self.MLP(X_count)
        count_truth = torch.sum(y_count, dim=1, keepdim=True)
        Loss_count = F.cross_entropy(count_predictions, (count_truth.squeeze() - 1).long(), reduction='mean')
        ########################Loss_count########################
        return Loss_sym / (2 * self.sigma1**2) + Loss_ce / (2 * self.sigma3**2) + Loss_count / (2 * self.sigma4**2) + torch.log(self.sigma1) + torch.log(self.sigma3) + torch.log(self.sigma4)
    

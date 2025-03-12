import torch
import torch.nn.functional as F

def sinkhorn_knopp(cost_matrix, epsilon=0.05, num_iters=50):
    """
    Sinkhorn-Knopp 算法实现最优传输分配
    Args:
        cost_matrix (Tensor): 代价矩阵 [B, C] (注意这里是负相似度 -S)
        epsilon (float): 熵正则化系数
        num_iters (int): 迭代次数
    Returns:
        Q (Tensor): 分配矩阵 [B, C]
    """
    # 初始化：使用指数缩放后的代价矩阵
    K = torch.exp(-cost_matrix / epsilon)
    
    # 交替行列归一化
    u = torch.ones(K.shape[0], device=K.device) / K.shape[0]  # 均匀分布约束
    v = torch.ones(K.shape[1], device=K.device) / K.shape[1]
    
    for _ in range(num_iters):
        # 行归一化
        K = K / u.view(-1, 1)  # [B, C]
        K = K / K.sum(dim=1, keepdim=True)  # 行和归一化为1
        
        # 列归一化
        K = K / v.view(1, -1)  # [B, C]
        K = K / K.sum(dim=0, keepdim=True)  # 列和归一化为1
    
    Q = K * K.shape[0]  # 缩放回原始比例（满足分配约束）
    return Q

def compute_unknown_alignment_loss(image_features, text_features, tau=0.07, epsilon=0.05):
    """
    计算未知类的对齐损失
    Args:
        image_features (Tensor): 未知类图像特征 [B, D]
        text_features (Tensor): 未知类文本特征 [C, D] (C=4)
        tau (float): 温度参数
        epsilon (float): Sinkhorn正则化系数
    Returns:
        loss (Tensor): 对比损失
        Q (Tensor): 分配矩阵 [B, C]
    """
    # Step 1: 计算图像-文本相似度矩阵 (余弦相似度)
    image_features = F.normalize(image_features, p=2, dim=1)  # [B, D]
    text_features = F.normalize(text_features, p=2, dim=1)  # [C, D]
    logits = torch.mm(image_features, text_features.t())  # [B, C]
    
    # Step 2: 最优传输分配 (将相似度转换为代价矩阵)
    cost_matrix = -logits  # Sinkhorn输入为代价矩阵，这里用负相似度
    Q = sinkhorn_knopp(cost_matrix, epsilon=epsilon)  # [B, C]
    
    # Step 3: 计算对比损失 (交叉熵形式)
    logits = logits / tau
    log_prob = F.log_softmax(logits, dim=1)  # [B, C]
    
    # 计算加权交叉熵损失
    loss = -torch.sum(Q * log_prob) / Q.size(0)
    
    return loss, Q

# 示例用法
if __name__ == "__main__":
    # 模拟数据
    batch_size = 64
    feat_dim = 512
    num_unknown_classes = 4
    
    # 随机生成特征 (实际应替换为真实特征)
    image_features = torch.randn(batch_size, feat_dim).cuda()
    unknown_texts = torch.randn(num_unknown_classes, feat_dim).cuda()
    
    # 计算损失
    loss, Q = compute_unknown_alignment_loss(
        image_features, 
        unknown_texts,
        tau=0.07,
        epsilon=0.05
    )
    
    print(f"Alignment Loss: {loss.item():.4f}")
    print(f"Assignment Matrix Q shape: {Q.shape}")
    print(f"Q column sums (should be ~{batch_size/num_unknown_classes}): {Q.sum(dim=0)}")
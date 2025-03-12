import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignModel(nn.Module):
    def __init__(self, text_features, known_classes, unknown_classes=4, tau=0.07):
        super().__init__()
        # 归一化文本特征（固定或可学习）
        self.register_buffer("text_features", F.normalize(text_features, dim=-1, p=2))
        self.tau = nn.Parameter(torch.tensor(tau))  # 可学习温度参数
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes

    def forward(self, image_features):
        # 归一化图像特征
        image_features = F.normalize(image_features, dim=-1, p=2)
        # 计算所有类别的相似度（已知类 + 未知类）
        logits = torch.matmul(image_features, self.text_features.T) / self.tau
        return logits  # [batch_size, total_classes]
    
def build_class_weights(class_freq, known_classes):
    # 已知类权重：与频率成反比
    known_weights = 1.0 / (class_freq[:known_classes] + 1e-8)
    known_weights = known_weights / known_weights.sum()  # 归一化
    # 未知类权重：假设均匀分布或自定义
    unknown_weights = torch.ones(4) / 4  
    return torch.cat([known_weights, unknown_weights])

# 假设已知类频率为 [0.6, 0.3, 0.1]，未知类均匀分布
class_freq = torch.tensor([0.6, 0.3, 0.1] + [0.25]*4)
weights = build_class_weights(class_freq, known_classes=3)
criterion = nn.CrossEntropyLoss(weight=weights)

def generate_pseudo_labels(model, unlabeled_features, threshold=0.8):
    model.eval()
    with torch.no_grad():
        logits = model(unlabeled_features)
        probs = F.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        mask = confidences > threshold
    return preds[mask], unlabeled_features[mask]  # 返回高置信度伪标签

# 示例：假设 unlabeled_features 是未知类的图像特征
pseudo_labels, pseudo_features = generate_pseudo_labels(model, unlabeled_features)

# 初始化模型和优化器
text_features = torch.randn(7, 512)  # 3已知类 +4未知类
model = AlignModel(text_features, known_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    # 训练已知类
    model.train()
    for image_features, labels in labeled_loader:  # 已知类数据
        logits = model(image_features)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 生成伪标签并加入训练
    pseudo_labels, pseudo_features = generate_pseudo_labels(model, unlabeled_features)
    if len(pseudo_labels) > 0:
        pseudo_logits = model(pseudo_features)
        pseudo_loss = criterion(pseudo_logits, pseudo_labels)
        optimizer.zero_grad()
        pseudo_loss.backward()
        optimizer.step()
        
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha  # 控制类别权重（如 alpha=0.25 对少数类加权）
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 替换原有损失函数
criterion = BalancedFocalLoss(alpha=0.25)
# 根据类别频率调整预测 logit
class_freq = torch.tensor([0.6, 0.3, 0.1] + [0.25]*4)  # 已知3类 + 未知4类
logit_bias = torch.log(class_freq + 1e-8)

def predict(logits, lamd=0.5):
    adjusted_logits = logits - lamd * logit_bias.to(logits.device)
    return torch.argmax(adjusted_logits, dim=1)

# 推理示例
image_features = torch.randn(10, 512)  # 假设提取的图像特征
logits = model(image_features)
preds = predict(logits)
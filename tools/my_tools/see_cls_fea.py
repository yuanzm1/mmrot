import pickle
# 指定pickle文件的路径，需要根据实际保存的位置修改
prefix = "dota_153638_"
file_path = f"/mnt/disk2/yuanzm/weights/{prefix}trans_features.pkl"
with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)
# print(loaded_dict)

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pdb

import torch


def cosine_similarity(a, b):
    """
    计算两个torch.Tensor类型的向量a和b的余弦相似度，考虑向量可能为0向量的情况
    """
    if torch.all(a == 0):
        if torch.all(b == 0):
            return 1.0  # 两个都是0向量，认为它们完全相似（余弦相似度为1）
        return 0.0  # a是0向量，b不是，认为它们完全不相似（余弦相似度为0）
    if torch.all(b == 0):
        return 0.0  # b是0向量，a不是，认为它们完全不相似（余弦相似度为0）

    dot_product = torch.sum(a * b)
    norm_a = torch.sqrt(torch.sum(a ** 2))
    norm_b = torch.sqrt(torch.sum(b ** 2))
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def calculate_class_means(selected_vectors_dict):
    """
    计算每类向量的平均值
    """
    class_means = {}
    for class_label, vectors in selected_vectors_dict.items():
        vectors_tensor = torch.stack(vectors)
        mean_vector = torch.mean(vectors_tensor, dim=0)
        class_means[class_label] = mean_vector
    return class_means

def calculate_similarities_between_class_means(class_means):
    """
    计算类别平均向量之间的相似度
    """
    num_classes = len(class_means)
    similarities = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            similarity = torch.nn.functional.cosine_similarity(class_means[i].unsqueeze(0), class_means[j].unsqueeze(0))
            similarities[i, j] = similarity
    return similarities

def plot_confusion_matrix(similarities, class_labels, save_path):
    """
    绘制带有相似度分数标注的混淆矩阵并保存
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(similarities.detach().cpu().numpy(), cmap='Blues')

    # 设置坐标轴标签、标题
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=6)  # 设置x轴标签字体大小为12
    ax.set_yticklabels(class_labels, fontsize=6)  # 设置y轴标签字体大小为12
    ax.set_xlabel('Predicted Label', fontsize=8)  # 设置x轴名称字体大小为14
    ax.set_ylabel('True Label', fontsize=8)  # 设置y轴名称字体大小为14
    
    
    # 在每个单元格中添加相似度分数标注
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, f"{similarities[i, j].item():.2f}",
                           ha="center", va="center", color="black", fontsize=10)  # 设置单元格内文字字体大小为10
    
    similarities_np = similarities.detach().cpu().numpy()
    ax.set_title(f'Similarity Means:{np.mean(similarities_np[similarities_np!=1])}', fontsize=10)  # 设置标题字体大小为16

    # 添加颜色条
    fig.colorbar(im)

    # 保存图像
    plt.savefig(save_path)
    plt.show()

def calculate_intra_class_similarity(selected_vectors_dict):
    """
    不使用循环计算类内相似度
    """
    intra_class_similarities = defaultdict(list)
    for class_label, vectors in selected_vectors_dict.items():
        vectors_tensor = torch.stack(vectors)
        num_vectors = vectors_tensor.shape[0]
        
        # 计算向量之间的点积，得到形状为 [num_vectors, num_vectors] 的矩阵
        dot_product_matrix = torch.mm(vectors_tensor, vectors_tensor.t())
        
        # 计算向量的模（范数），得到形状为 [num_vectors] 的向量
        norm_vectors = torch.sqrt(torch.sum(vectors_tensor ** 2, dim=1))
        
        # 通过广播机制，将范数向量扩展为形状为 [num_vectors, num_vectors] 的矩阵，用于计算余弦相似度
        norm_matrix = torch.ger(norm_vectors, norm_vectors)
        
        # 计算余弦相似度矩阵，通过点积矩阵除以范数矩阵
        similarity_matrix = dot_product_matrix / norm_matrix
        
        # 提取上三角部分（排除对角线元素，因为自己与自己的相似度为1没必要重复计算）
        upper_tri_mask = torch.triu(torch.ones(num_vectors, num_vectors, dtype=torch.bool).to('cuda'), diagonal=1)
        upper_tri_similarities = similarity_matrix[upper_tri_mask]
        
        # 将该类的相似度值添加到对应字典中
        intra_class_similarities[class_label].extend(upper_tri_similarities.tolist())
    
    return intra_class_similarities

def calculate_inter_class_similarity(selected_vectors_dict):
    """
    不使用循环计算类间相似度
    """
    inter_class_similarities = []
    class_labels = list(selected_vectors_dict.keys())
    all_vectors = [torch.stack(vectors) for vectors in selected_vectors_dict.values()]
    
    for i in range(len(class_labels)):
        for j in range(i + 1, len(class_labels)):
            vectors_i = all_vectors[i]
            vectors_j = all_vectors[j]
            
            # 计算两个不同类别向量之间的点积，得到形状为 [num_vectors_i, num_vectors_j] 的矩阵
            dot_product_matrix = torch.mm(vectors_i, vectors_j.t())
            
            # 计算两个不同类别向量各自的模（范数），分别得到形状为 [num_vectors_i] 和 [num_vectors_j] 的向量
            norm_vectors_i = torch.sqrt(torch.sum(vectors_i ** 2, dim=1))
            norm_vectors_j = torch.sqrt(torch.sum(vectors_j ** 2, dim=1))
            
            # 通过广播机制扩展范数向量，用于计算余弦相似度
            norm_matrix_i = torch.ger(norm_vectors_i, torch.ones(len(vectors_j)).to('cuda'))
            norm_matrix_j = torch.ger(torch.ones(len(vectors_i)).to('cuda'), norm_vectors_j)
            norm_matrix = norm_matrix_i * norm_matrix_j
            
            # 计算余弦相似度矩阵，通过点积矩阵除以范数矩阵
            similarity_matrix = dot_product_matrix / norm_matrix
            
            # 将该部分相似度值添加到总的类间相似度列表中
            inter_class_similarities.extend(similarity_matrix.view(-1).tolist())
    
    return inter_class_similarities

selected_vectors_dict = defaultdict(list)
for class_label, vectors in loaded_dict.items():
    if len(vectors) >= 100:
        selected_indices = random.sample(range(len(vectors)), 100)
        for index in selected_indices:
            selected_vectors_dict[class_label].append(vectors[index])
    else:
        print(f"类别 {class_label} 的向量数量不足100个，无法进行完整选取")
        
selected_vectors_dict = dict(sorted(selected_vectors_dict.items()))
                
# 假设已经通过上述函数计算得到了类内相似度和类间相似度结果，以下是画图展示并保存的代码
# 绘制类内相似度箱线图
intra_class_similarities = calculate_intra_class_similarity(selected_vectors_dict)
intra_class_similarities = dict(sorted(intra_class_similarities.items()))
# pdb.set_trace()
plt.figure(figsize=(10, 6))
plt.boxplot([intra_class_similarities[label] for label in intra_class_similarities], labels=list(intra_class_similarities.keys()))
plt.xlabel('Class Label')
plt.ylabel('Cosine Similarity')
plt.title('Intra-class Cosine Similarity')
# 保存类内相似度图
plt.savefig(f'{prefix}intra.png')

# 绘制类间相似度直方图
inter_class_similarities = calculate_inter_class_similarity(selected_vectors_dict)
plt.figure(figsize=(10, 6))
plt.hist(inter_class_similarities, bins=20, edgecolor='black')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Inter-class Cosine Similarity')
# 保存类间相似度图
plt.savefig(f'{prefix}inter.png')

plt.show()

class_means = calculate_class_means(selected_vectors_dict)
similarities = calculate_similarities_between_class_means(class_means)
class_labels = list(selected_vectors_dict.keys())
save_path = f"{prefix}class_mean_sim.png"
plot_confusion_matrix(similarities, class_labels, save_path)
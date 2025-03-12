import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pdb
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 示例数据生成
import torch
import torch.nn as nn
nwpu = ['bri', 'base','ship', 'bask', 'har', 
                            'gtf','ap','st','vehi','ten']
dior = ['ape', 'base', 'bri', 'chim','dam','ESa','Ets', 'golf', 'har', 
        'ovp', 'ship', 'stad', 'st', 'tennis', 'ts', 'vehi', 'apt','basket','gtf','wm']
dota = ['pla','shi','stor','base','bask','gtf','har','bri','l-v',
               's-v','round','heli', 'ten',  'soc',  'swim']
cls_names = dior
# feature = torch.randn(n, 1024).numpy()
# labels = torch.randint(0, 5, (n,)).numpy()

# feature = torch.load(r'G:\project\in-class\mmpre\my_tools\raw_features.pth').cpu().numpy()
# labels = torch.load(r'G:\project\in-class\mmpre\my_tools\label.pth').cpu().numpy()
name = 'diorbest'
num_cls = len(cls_names)
feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_feats.npy')
labels = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_labels.npy')
text_feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_textfs.npy')
# pdb.set_trace()

# # 计算未知类的准确分数
mask = labels > (num_cls-4)
feature = feature[mask]
true_labels = torch.from_numpy(labels[mask] - (num_cls-4) - 1)
ori_text, align_text = text_feature[:num_cls,:], text_feature[num_cls:, :]
align_text = F.normalize(torch.from_numpy(align_text), dim=1)
feature = F.normalize(torch.from_numpy(feature), dim=1)
res = torch.matmul(feature, align_text.transpose(-1, -2))
res2 = torch.matmul(align_text, align_text.transpose(-1, -2))
softmax_layer = nn.Softmax(dim = 1)
unknown_res = softmax_layer(res[:,(num_cls-4):])
pred_labels = torch.argmax(unknown_res, dim = 1)
cls_names = cls_names[-4:]

# # # 计算已知类的准确分数
# mask = labels <= (num_cls-4)
# feature = feature[mask]
# true_labels = torch.from_numpy(labels[mask]) - 1
# ori_text, align_text = text_feature[:num_cls,:], text_feature[num_cls:, :]
# align_text = F.normalize(torch.from_numpy(align_text), dim=1)
# feature = F.normalize(torch.from_numpy(feature), dim=1)
# res = torch.matmul(feature, align_text.transpose(-1, -2))
# softmax_layer = nn.Softmax(dim = 1)
# unknown_res = softmax_layer(res[:,:(num_cls-4)])
# pred_labels = torch.argmax(unknown_res, dim = 1)

# 计算所有类的准确分数
# true_labels = torch.from_numpy(labels) - 1
# ori_text, align_text = text_feature[:num_cls,:], text_feature[num_cls:, :]
# align_text = F.normalize(torch.from_numpy(align_text), dim=1)
# feature = F.normalize(torch.from_numpy(feature), dim=1)
# res = torch.matmul(feature, align_text.transpose(-1, -2))
# softmax_layer = nn.Softmax(dim = 1)
# unknown_res = softmax_layer(res)
# pred_labels = torch.argmax(unknown_res, dim = 1)

# print(res[:10], true_labels[:10])
# print("res2", res2)
# 比较预测标签和真实标签，得到一个布尔型张量
correct_predictions = pred_labels == true_labels
# 统计预测正确的样本数量
num_correct = correct_predictions.sum().item()
# 计算总样本数量
total_samples = len(true_labels)
# 计算预测准确度
accuracy = num_correct / total_samples
print(f"预测准确度: {accuracy * 100:.2f}%")

# 计算混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
            xticklabels=cls_names, yticklabels=cls_names)
plt.xlabel('Predicted Labels', color='red')
plt.ylabel('True Labels', color='red')
plt.title('Confusion Matrix')
plt.savefig(f'tools/my_tools/{name}_confusion_matrix.png')


# # 使用np.unique函数，设置return_counts为True
# unique_labels, counts = np.unique(true_labels, return_counts=True)
# # 输出结果
# for label, count in zip(unique_labels, counts):
#     print(f"标签 {label} 的个数是 {count}")





#####################################################################################
# labels = np.arange(10)
# tsne = TSNE(n_components=2, random_state=42, perplexity=9)
# feature_2d = tsne.fit_transform(align_text)

# num_classes = len(np.unique(labels))
# cmap = plt.get_cmap('tab10', num_classes)
# # cmap = plt.get_cmap('viridis', num_classes)
# #pdb.set_trace()

# # 可视化t-SNE结果
# plt.figure(figsize=(10, 8))
# # 绘制普通向量，根据类别标签着色
# scatter = plt.scatter(feature_2d[:, 0], feature_2d[:, 1], c=labels, cmap=cmap)

# # 获取legend_elements返回的句柄和默认标签
# handles, _ = scatter.legend_elements()
# # 使用具体类别名称替换默认数字标签
# legend1 = plt.legend(handles, cls_names, title="Classes", loc='upper left')
# plt.gca().add_artist(legend1)

# # 为原型向量添加单独的图例
# legend2 = plt.legend([plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=10)], ['Prototype Vectors'], loc='upper right')
# plt.gca().add_artist(legend2)

# # 添加标题和轴标签
# plt.title('t-SNE Visualization of Features')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')

# # plt.savefig('tools/my_tools/dior_bf_tsne.png')
# # plt.savefig('tools/my_tools/dior_tsne.png')
# plt.savefig(f'tools/my_tools/try.png')
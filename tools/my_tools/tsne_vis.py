import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pdb

# 示例数据生成
import torch
nwpu = ['bridge', 'baseball','ship', 'basketball', 'harbor', 
                            'groundtrackfield','airplane','storagetank','vehicle','tenniscourt']
dior = ['airplane', 'baseballfield', 'bridge', 'chimney','dam','Expressway-Service-area','Expressway-toll-station', 'golffield', 'harbor', 
        'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'airport','basketballcourt','groundtrackfield','windmill']
dota = ['plane','ship','storage','baseball','basketball','gtf','harbor','bridge','large-vehicle',
               'small-vehicle','roundabout','helicopter', 'tennis',  'soccer',  'swimming']
cls_names = dota
# n = 100
# feature = torch.randn(n, 1024).numpy()
# labels = torch.randint(0, 5, (n,)).numpy()

# feature = torch.load(r'G:\project\in-class\mmpre\my_tools\raw_features.pth').cpu().numpy()
# labels = torch.load(r'G:\project\in-class\mmpre\my_tools\label.pth').cpu().numpy()
name = 'dota1'
feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_feats.npy')
labels = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_labels.npy')
text_feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/dota2_textfs.npy')
# text_feature = np.random.rand(30,1024)
# mask = labels < len(cls_names) - 5
# feature = feature[mask]
# labels = labels[mask]    #- (len(cls_names) - 5)
# cls_names = cls_names[len(cls_names) - 5:]
ori_text, align_text = text_feature[:len(cls_names),:], text_feature[len(cls_names):, :]
#align_text = F.normalize(torch.from_numpy(align_text), dim=1).numpy()
n_uk = 15
uk_align_text = ori_text
# uk_align_text = align_text[-n_uk:,:]

##############################################################
# # 获取唯一的label值
# unique_labels = np.unique(labels)
# print(unique_labels)
# # 用于存储每个label对应的feature的平均值
# average_features = []
# for label in unique_labels:
#     # 找到所有对应label的索引
#     label_indices = np.where(labels == label)[0]
#     # 计算这些索引对应的feature的平均值
#     avg_feature = np.mean(feature[label_indices], axis=0)
#     average_features.append(avg_feature)
# stacked_features = np.stack(average_features, axis=0)
# n_uk = 15
# uk_align_text = stacked_features
##############################################################

# 每个标签最多选取 1000 个特征
max_samples_per_label = 500
selected_features = []
selected_labels = []
# 获取唯一的标签
unique_labels = np.unique(labels)
for label in unique_labels:
    # 找出属于当前标签的特征索引
    label_indices = np.where(labels == label)[0]
    # 如果该标签的特征数量超过 1000 个，则随机选取 1000 个
    if len(label_indices) > max_samples_per_label:
        selected_indices = np.random.choice(label_indices, max_samples_per_label, replace=False)
    else:
        selected_indices = label_indices
    # 将选取的特征和标签添加到列表中
    selected_features.extend(feature[selected_indices])
    selected_labels.extend(labels[selected_indices])
# 将列表转换为 numpy 数组
selected_features = np.array(selected_features)
selected_labels = np.array(selected_labels)
feature = selected_features
labels = selected_labels

# pdb.set_trace()
# 合并所有向量
all_vectors = np.vstack((feature, uk_align_text))
# pdb.set_trace()
# 使用t-SNE进行降维
#tsne = TSNE(n_components=2, perplexity=100, max_iter=5000, learning_rate=200, metric='cosine')
tsne = TSNE(n_components=2, random_state=42)
feature_2d_all = tsne.fit_transform(all_vectors)
feature_2d = feature_2d_all[:-n_uk]
reduced_vectors_prototype = feature_2d_all[-n_uk:]

# 创建一个颜色映射，为每个标签分配一种颜色
num_classes = len(np.unique(labels))
cmap = plt.get_cmap('tab20', num_classes)
# cmap = plt.get_cmap('viridis', num_classes)
#pdb.set_trace()

# 可视化t-SNE结果
plt.figure(figsize=(10, 8))
# 绘制普通向量，根据类别标签着色
scatter = plt.scatter(feature_2d[:, 0], feature_2d[:, 1], c=labels, cmap=cmap)

# 绘制原型向量并着重标注
for i, (x, y) in enumerate(reduced_vectors_prototype):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
plt.scatter(reduced_vectors_prototype[:, 0], reduced_vectors_prototype[:, 1], c='r', marker='*', s=200, label='Prototype Vectors')

# 获取legend_elements返回的句柄和默认标签
handles, _ = scatter.legend_elements()
# 使用具体类别名称替换默认数字标签
legend1 = plt.legend(handles, cls_names, title="Classes", loc='upper left')
plt.gca().add_artist(legend1)

# 为原型向量添加单独的图例
legend2 = plt.legend([plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=10)], ['Prototype Vectors'], loc='upper right')
plt.gca().add_artist(legend2)

# 添加标题和轴标签
plt.title('t-SNE Visualization of Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# plt.savefig('tools/my_tools/dior_bf_tsne.png')
# plt.savefig('tools/my_tools/dior_tsne.png')
plt.savefig(f'tools/my_tools/{name}.png')
# 显示图形
# plt.show()
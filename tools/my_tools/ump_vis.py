import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pdb

import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt

nwpu = ['bridge', 'baseball','ship', 'basketball', 'harbor', 
                            'groundtrackfield','airplane','storagetank','vehicle','tenniscourt']
dior = ['airplane', 'baseballfield', 'bridge', 'chimney','dam','Expressway-Service-area','Expressway-toll-station', 'golffield', 'harbor', 
        'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'airport','basketballcourt','groundtrackfield','windmill']
dota = ['plane','ship','storage','baseball','basketball','gtf','harbor','bridge','large-vehicle',
               'small-vehicle','roundabout','helicopter', 'tennis',  'soccer',  'swimming']
cls_names = nwpu
# n = 100
# feature = torch.randn(n, 1024).numpy()
# labels = torch.randint(0, 5, (n,)).numpy()

# feature = torch.load(r'G:\project\in-class\mmpre\my_tools\raw_features.pth').cpu().numpy()
# labels = torch.load(r'G:\project\in-class\mmpre\my_tools\label.pth').cpu().numpy()
name = 'nwpubest2'
feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_feats.npy')
labels = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_labels.npy')
text_feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_textfs.npy')
# pdb.set_trace()
# mask = labels > len(cls_names) - 10
# feature = feature[mask]
# labels = labels[mask] - (len(cls_names) - 10)
# cls_names = cls_names[len(cls_names) - 10:]
ori_text, align_text = text_feature[:10,:], text_feature[10:, :]
#align_text = F.normalize(torch.from_numpy(align_text), dim=1).numpy()
n_uk = 10
uk_align_text = align_text[-n_uk:,:]
#feature = F.normalize(torch.from_numpy(feature), dim=1).numpy()
# uk_align_text = np.random.randn(4,1024)
# pdb.set_trace()

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

image_feats = feature
semantic_feats = align_text
# 假设labels是一个numpy数组，存储图像特征的类别
# 假设semantic_labels是长度为n_cls的列表，存储语义特征的类别
labels = np.array([f'Image_{i % 5}' for i in range(image_feats.shape[0])])  # 示例图像标签，假设有5个类别
semantic_labels = [f'Semantic_{i}' for i in range(semantic_feats.shape[0])]  # 示例语义标签

# 对图像标签进行编码，以便映射到颜色
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

combined_feats = np.concatenate([semantic_feats, image_feats], axis=0)
reducer = UMAP(n_neighbors=5, min_dist=0.3)
embedding = reducer.fit_transform(combined_feats)

# 可视化
plt.figure(figsize=(10, 6))

# 绘制语义特征点并标记文字
for i, label in enumerate(semantic_labels):
    plt.scatter(embedding[i, 0], embedding[i, 1], c='black', marker='x')
    plt.annotate(label, xy=(embedding[i, 0], embedding[i, 1]), xytext=(5, 5), textcoords='offset points')

# 绘制图像特征点并按颜色标记
for i in range(len(image_feats)):
    plt.scatter(embedding[len(semantic_feats) + i, 0], embedding[len(semantic_feats) + i, 1],
                c=plt.cm.tab10(encoded_labels[i]), label=f'Image_{labels[i]}' if i == 0 else "")

plt.legend()
plt.show()
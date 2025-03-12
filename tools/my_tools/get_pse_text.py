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
cls_names = nwpu
# n = 100
# feature = torch.randn(n, 1024).numpy()
# labels = torch.randint(0, 5, (n,)).numpy()

# feature = torch.load(r'G:\project\in-class\mmpre\my_tools\raw_features.pth').cpu().numpy()
# labels = torch.load(r'G:\project\in-class\mmpre\my_tools\label.pth').cpu().numpy()
name = 'dota1'
feature = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_feats.npy')
labels = np.load(f'/mnt/disk2/yuanzm/weights/mmrot_weights/{name}_labels.npy')

# 获取唯一的label值
unique_labels = np.unique(labels)
print(unique_labels)
# 用于存储每个label对应的feature的平均值
average_features = []

for label in unique_labels:
    # 找到所有对应label的索引
    label_indices = np.where(labels == label)[0]
    # 计算这些索引对应的feature的平均值
    avg_feature = np.mean(feature[label_indices], axis=0)
    average_features.append(avg_feature)

# 将平均特征列表堆叠成一个二维数组
stacked_features = np.stack(average_features, axis=0)
a = torch.load('/home/yuanzm/mmpretrain-main/weights/leak4-dota_dsp.pth')
print(a.shape)

b = torch.from_numpy(stacked_features)
torch.save(b, "/home/yuanzm/mmpretrain-main/weights/dota_pse.pth")
print("堆叠后的数组形状:", b.shape)
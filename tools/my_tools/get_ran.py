import torch

a = torch.randn(15,1024)
torch.save(a, '/home/yuanzm/mmpretrain-main/weights/dota_rand.pth')
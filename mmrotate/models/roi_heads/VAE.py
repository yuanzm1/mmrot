import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

# 定义编码器（对于视觉和文本）
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


# 定义解码器（对于视觉和文本）
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        x = self.sigmoid(self.fc2(z))
        return x


# 定义双路 VAE 结构
class DualVAE(nn.Module):
    def __init__(self, input_size_visual, input_size_text, hidden_size, latent_size, num_classes):
        super(DualVAE, self).__init__()
        self.encoder_visual = Encoder(input_size_visual, hidden_size, latent_size)
        self.encoder_text = Encoder(input_size_text, hidden_size, latent_size)
        self.decoder_visual = Decoder(latent_size, hidden_size, input_size_visual)
        self.decoder_text = Decoder(latent_size, hidden_size, input_size_text)
        self.aux_cls_visual = LINEAR_LOGSOFTMAX(latent_size, num_classes)
        self.aux_cls_text = LINEAR_LOGSOFTMAX(latent_size, num_classes)
        self.latent_size = latent_size
        self.reparameterize_with_noise = True
        
    def reparameterize(self, mean, logvar):
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # z = mean + eps * std
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mean + sigma * eps
        else:
            return mean
    
    def add_cross_weight(self, img, att):
        Q = img.view(-1, self.latent_size, 1)
        K = att.view(-1, 1, self.latent_size)
        R = torch.bmm(Q, K)
        soft_R = F.softmax(R, 1)
        _img = torch.bmm(soft_R, img.unsqueeze(2)).squeeze()
        _att = torch.bmm(soft_R, att.unsqueeze(2)).squeeze()

        return _img, _att

    def forward(self, x_visual, x_text):
        if self.reparameterize_with_noise:
            # 视觉分支
            mean_visual, logvar_visual = self.encoder_visual(x_visual)
            z_visual = self.reparameterize(mean_visual, logvar_visual)
            x_visual_recon = self.decoder_visual(z_visual)

            # 文本分支
            mean_text, logvar_text = self.encoder_text(x_text)
            z_text = self.reparameterize(mean_text, logvar_text)
            x_text_recon = self.decoder_text(z_text)
            
            z_visual_, z_text_ = self.add_cross_weight(z_visual, z_text)
            out_v = self.aux_cls_visual(z_visual_)
            out_s = self.aux_cls_text(z_text_)
            
            return x_visual_recon, mean_visual, logvar_visual, x_text_recon, mean_text, logvar_text, out_v, out_s
        else:
            mean_visual, logvar_visual = self.encoder_visual(x_visual)
            z_visual = self.reparameterize(mean_visual, logvar_visual)
            
            mean_text, logvar_text = self.encoder_text(x_text)
            z_text = self.reparameterize(mean_text, logvar_text)
            
            return z_visual, z_text


# 损失函数：重构损失 + KL 散度损失
def loss_function_vae(recon_x, x, mean, logvar, factor=3):
    BCE = nn.L1Loss(size_average=False)(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE/100 + KLD


# # 超参数设置
# input_size_visual = 784  # 例如对于 28x28 的图像
# input_size_text = 100  # 例如文本特征的维度
# hidden_size = 256
# latent_size = 128
# learning_rate = 1e-3
# epochs = 100
# batch_size = 128


# # 初始化模型和优化器
# model = DualVAE(input_size_visual, input_size_text, hidden_size, latent_size)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
#                         torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
# distance = distance.sum()

# 假设我们有数据加载器（这里仅为示例，需要根据实际情况创建）
# train_loader_visual 和 train_loader_text 分别是视觉和文本数据的加载器
# for epoch in range(epochs):
#     for x_visual, x_text in zip(train_loader_visual, train_loader_text):
#         # 前向传播
#         x_visual_recon, mean_visual, logvar_visual, x_text_recon, mean_text, logvar_text = model(x_visual, x_text)

#         # 计算损失
#         loss_visual = loss_function(x_visual_recon, x_visual, mean_visual, logvar_visual)
#         loss_text = loss_function(x_text_recon, x_text, mean_text, logvar_text)
#         loss = loss_visual + loss_text

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# 代码解释：
# 1. Encoder 类：
#    - 接收输入，通过线性层和 ReLU 激活，将输入映射到隐层，然后分别计算均值和对数方差。
# 2. Decoder 类：
#    - 接收来自潜在空间的向量，通过线性层和激活函数，将其映射回输入空间。
# 3. DualVAE 类：
#    - 包含视觉和文本的编码器和解码器，在 forward 方法中，分别对视觉和文本进行编码、重参数化和解码操作。
# 4. loss_function 函数：
#    - 计算重构损失（使用 BCE）和 KL 散度损失，两者相加得到总损失。
# 5. 超参数：
#    - 定义了输入、隐藏层、潜在层的维度，学习率和训练的轮数等。
# 6. 模型和优化器：
#    - 初始化 DualVAE 模型和 Adam 优化器。
# 7. 训练部分：
#    - 虽然这里没有完整的训练代码，但提供了一个基本的训练框架，使用 zip 同时处理视觉和文本数据，计算损失并更新模型。
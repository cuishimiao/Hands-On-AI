import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 自动检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 超参数配置
LATENT_DIM = 100
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 100

# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, IMG_SIZE**2),
            nn.Tanh()  # 输出范围[-1,1]
        )
    
    def forward(self, z):
        return self.main(z).view(-1, 1, IMG_SIZE, IMG_SIZE)

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(IMG_SIZE**2, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(-1, IMG_SIZE**2)
        return self.main(img_flat)

# 初始化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 优化器配置
g_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 数据加载（MNIST示例）
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),  # 启用3通道加速
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
])

dataset = datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(
    dataset,
    num_workers=min(4, os.cpu_count()//2),  # 不超过CPU核心半数
    prefetch_factor=2,  # 降低预取批次
    persistent_workers=False,  # MPS暂不支持持久化
    multiprocessing_context='spawn'  # 强制使用spawn模式
)


# 损失函数
criterion = nn.BCELoss()

# 固定噪声用于生成示例
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

for epoch in range(EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 在训练循环中定期释放缓存
        if device.type == 'mps':
            torch.mps.empty_cache()
        
        # 训练判别器
        d_optim.zero_grad()
        
        # 真实图片
        real_labels = torch.ones(batch_size, 1, device=device)
        d_real_loss = criterion(discriminator(real_imgs), real_labels)
        
        # 生成图片
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = generator(noise).detach()
        fake_labels = torch.zeros(batch_size, 1, device=device)
        d_fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optim.step()
        
        # 训练生成器
        g_optim.zero_grad()
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_imgs = generator(noise)
        g_loss = criterion(discriminator(gen_imgs), real_labels)  # 欺骗判别器
        g_loss.backward()
        g_optim.step()
        
    # 每10个epoch生成示例
    if epoch % 10 == 0:
        with torch.no_grad():
            gen_imgs = generator(fixed_noise)
            plt.figure(figsize=(8,8))
            for j in range(16):
                plt.subplot(4,4,j+1)
                plt.imshow(gen_imgs[j].cpu().squeeze(), cmap='gray')
                plt.axis('off')
            plt.savefig(f'generated_epoch{epoch}.png')
            plt.close()
            
        print(f'Epoch [{epoch}/{EPOCHS}] | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}')

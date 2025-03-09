import torchvision
from torch.utils.data import DataLoader

# 1. 数据加载优化（启用Metal加速）
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # Metal加速的随机裁剪
    torchvision.transforms.RandomCrop(224)
])

# 2. 模型配置（自动切换设备）
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device)
model.train()

# 3. 训练循环（带Metal优化）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for batch_idx, (data, target) in enumerate(DataLoader(dataset, batch_size=64)):
    data, target = data.to(device), target.to(device)
    
    with torch.autocast(device_type='mps', dtype=torch.float16):  # 混合精度
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
    
    optimizer.zero_grad(set_to_none=True)  # 内存优化
    loss.backward()
    optimizer.step()

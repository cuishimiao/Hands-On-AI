import torch

# 自动选择设备（优先使用MPS）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 显式启用Metal后端
torch.backends.mps.enabled = True
torch.backends.mps.is_available = lambda: True

# 设置内存增长策略（避免OOM）
if device.type == 'mps':
    torch.mps.set_per_process_memory_fraction(0.5)  # 限制内存使用比例
    torch.mps.empty_cache()  # 手动清理缓存


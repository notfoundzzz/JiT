import os
from PIL import Image
import numpy as np

# 建立模拟 ImageNet 的目录结构
train_path = "toy_data/train/class_0"
val_path = "toy_data/val/class_0"
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# 生成 10 张图片
for i in range(10):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(os.path.join(train_path, f"fake_{i}.jpg"))
    Image.fromarray(img).save(os.path.join(val_path, f"fake_{i}.jpg"))

print("✅ 模拟数据集已就绪：toy_data/")

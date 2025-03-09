# 导入必要库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import albumentations as A

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备(GPU或CPU，但是在线只能使用cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义KITTI深度数据集
class KITTIDepthDataset(Dataset):
    def __init__(self, data_dir, cam, split='train', val_length=10, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 加载数据路径
        self.image_paths = []
        self.depth_paths = []
        self.cam = cam
        imgs_lst = os.listdir(os.path.join(data_dir, self.cam, 'data'))

        if split == 'val':
            imgs_lst = imgs_lst[:val_length]
        self.image_paths = [os.path.join(data_dir, self.cam, 'data', i) for i in imgs_lst if i.endswith('.png')]
        self.image_paths.sort()
        self.depth_paths = [os.path.join(data_dir, 'proj_depth/groundtruth', self.cam, i) for i in imgs_lst if
                            i.endswith('.png')]
        self.depth_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取深度图（KITTI使用16位PNG存储）
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 256.0  # 转换为实际米数

        # 数据增强
        if self.transform:
            transformed = self.transform(image=image, depth=depth)
            # print(transformed_image)
            image = transformed["image"]
            depth = transformed["depth"]

        # 转换为Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()  # 增加通道维度
        # print(image.shape)
        # 深度归一化（假设最大深度80米）
        depth = torch.clamp(depth, 0.0, 80.0) / 80.0

        return image, depth


# 定义数据增强方法
train_transform = A.Compose([
    A.Resize(256, 512),  # KITTI原始分辨率为375x1242，调整至256*512
    # A.HorizontalFlip(p=0.5),  # 依概率进行水平翻转
    # A.ColorJitter(p=0.2),  # 依概率进行颜色抖动
], additional_targets={'depth': 'image'})

val_transform = A.Compose([
    A.Resize(256, 512),  # 只进行尺寸调整
], additional_targets={'depth': 'image'})

# 创建数据集
train_dataset = KITTIDepthDataset(
    data_dir='kitti/2011_09_26_drive_0001_sync',
    cam='image_02',
    split='train',
    transform=train_transform
)

val_dataset = KITTIDepthDataset(
    data_dir='kitti/2011_09_26_drive_0001_sync',
    cam='image_03',  # 和image_02是KITTI采集数据时不同相机输出的图像
    split='val',
    transform=val_transform
)


# 定义模型解码器网络层
class Decoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_layer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 输入x的维度是 (B, Cin, H, W)
        x = self.upsample(x)  # 使用双线性插值将输入特征图的空间尺寸放大两倍，维度变为 (B, Cin, 2*H, 2*W)
        x_skip = self.conv_skip(x)  # 残差分支不改变尺寸，只改变通道数，维度变为 (B, Cout, 2*H, 2*W)

        out = self.conv1(x)  # 卷积核3填充1，只有通道数发生变化，维度是(B, Cout, 2*H, 2*W)
        out = self.bn1(out)  # 归一化，不改变维度
        out = self.relu(out)  # 激活函数，不改变维度
        out = self.conv2(out)  # 卷积核3填充1，只有通道数发生变化，仍然是(B, Cout, 2*H, 2*W)
        out = self.bn2(out)  # 归一化，不改变维度
        out = self.relu(out)  # 激活函数，不改变维度

        return out + x_skip  # 残差连接


# 定义模型
class FCRN(nn.Module):
    def __init__(self):
        super(FCRN, self).__init__()
        # FCRN使用ResNet50作为编码器
        # 直接使用预训练的ResNet50模型
        # ResNet50 = models.resnet50(pretrained=True) # 联网下载预训练ResNet50
        ResNet50 = models.resnet50(pretrained=False)
        ResNet50.load_state_dict(torch.load("assets/resnet50-0676ba61.pth"))
        self.encoder = nn.Sequential(*list(ResNet50.children())[:-2])

        # 使用自定义小卷积上采样解码器
        self.up1 = Decoder_layer(2048, 1024)
        self.up2 = Decoder_layer(1024, 512)
        self.up3 = Decoder_layer(512, 256)
        self.up4 = Decoder_layer(256, 128)
        self.up5 = Decoder_layer(128, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出0-1之间的归一化深度
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        # return self.final(x) * 80.0  # 还原到0-80米范围
        return self.final(x)  # 最后的输出是0~1的归一化深度


# 创建模型
FCRN_Model = FCRN().to(device)
# print(FCRN_Model)

# 创建数据加载器
batch_size = 16  # 本地一次可以训练多张图片
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(val_dataset)}")

print(f"训练总批次: {len(train_loader)}")
print(f"验证总批次: {len(val_loader)}")


# 损失函数（BerHu Loss）
class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        max_diff = self.threshold * torch.max(diff).item()
        """
        如果 diff 小于或等于 max_diff，则 loss 等于 diff。
        如果 diff 大于 max_diff，则 loss 等于 (diff**2 + max_diff**2) / (2 * max_diff + 1e-6)。
        这种处理方式的目的是避免 diff 过大时对损失值产生过大的影响。
        """
        loss = torch.where(diff <= max_diff, diff, (diff ** 2 + max_diff ** 2) / (2 * max_diff + 1e-6))
        return torch.mean(loss)


criterion = BerHuLoss()

# 优化器
optimizer = optim.Adam(FCRN_Model.parameters(), lr=1e-4)

# 学习率调度器，使用余弦退火策略，并在指定的周期后重新启动学习率。其主要目的是在训练过程中周期性地调整学习率，以帮助模型跳出局部最优解并更好地收敛。
# T_0=10 表示第一个周期的长度为 10 个 epoch。T_mult=2 表示每个周期的长度按倍数增长（第二个周期为 20 个 epoch，第三个周期为 40 个 epoch，依此类推）。
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 定义训练轮数，可以根据需要调整
num_epochs = 10
best_rmse = float('inf')

for epoch in range(num_epochs):
    # 训练阶段
    FCRN_Model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
    # 获取数据并转移到cpu或cuda
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 正向传播与反向传播
        optimizer.zero_grad()
        outputs = FCRN_Model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    # 训练完成一回合，进行学习率退火
    scheduler.step()

    # 验证阶段
    FCRN_Model.eval()
    val_loss = 0.0
    rmse_values = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = FCRN_Model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 测试时，采用RMSE作为评价指标
            valid_mask = targets > 0.0
            rmse = torch.sqrt(((outputs[valid_mask] - targets[valid_mask]) ** 2).mean())
            rmse_values.append(rmse.item())

    avg_rmse = np.mean(rmse_values)
    print(f'Epoch {epoch + 1}/{num_epochs} | '
          f'Train Loss: {train_loss / len(train_loader):.4f} | '
          f'Val Loss: {val_loss / len(val_loader):.4f} | '
          f'RMSE: {avg_rmse:.2f}m')

    # 保存最佳模型
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        torch.save({
            'epoch': epoch,
            'model_state_dict': FCRN_Model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_rmse': best_rmse,
        }, 'best_model_kitti.pth')
        print(f"New best model saved with RMSE: {best_rmse:.2f}")

    # 保存最近一次的模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': FCRN_Model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rmse': avg_rmse,
    }, 'last_epoch_model_kitti.pth')


def visualize_kitti_results(model, dataset, batchsize=3):
    visualize_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(visualize_loader))
        inputs = inputs.to(device)
        outputs = model(inputs)

        inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
        targets = targets.cpu().numpy().squeeze()
        outputs = outputs.cpu().numpy().squeeze()

        plt.figure(figsize=(18, 8))
        for i in range(batchsize):
            # 输入图像
            plt.subplot(3, batchsize, i + 1)
            plt.imshow(inputs[i])
            plt.title('Input Image')
            plt.axis('off')

            # 真实深度
            plt.subplot(3, batchsize, 4 + i)
            plt.imshow(targets[i], cmap='jet', vmax=1)
            plt.title('Ground Truth')
            plt.axis('off')

            # 预测深度
            plt.subplot(3, batchsize, 7 + i)
            plt.imshow(outputs[i], cmap='jet', vmax=1)
            plt.title('Prediction')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# 加载最佳模型
checkpoint = torch.load('best_model_kitti.pth', map_location=device)
# 加载最新模型
# checkpoint = torch.load('last_epoch_model_kitti.pth', map_location=device)
# 加载预训练模型
# checkpoint = torch.load('pretrained_model.pth', map_location=device)

FCRN_Model.load_state_dict(checkpoint['model_state_dict'])

visualize_kitti_results(FCRN_Model, val_dataset, batchsize=3)

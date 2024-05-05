import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from astropy.io import fits
import numpy as np

# 生成波长-通量图像的函数
def create_wavelength_flux_image(flux, wavelength, repeat=10):
    # 将一维光谱数据转换为二维图像，减少重复次数以减小数组大小
    image = np.tile(flux.astype(np.float32), (repeat, 1))  # 使用32位浮点数
    return image

# 修改后的数据集类
class SpectraImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.images = []
        self.labels = []
        for file_name in file_list:
            with fits.open(file_name) as hdulist:
                flux = hdulist[0].data
                label = hdulist[1].data['label']
                wavelength = np.linspace(3900, 9000, len(flux))
                image = create_wavelength_flux_image(flux, wavelength)
                self.images.append(image)
                self.labels.append(label)
        self.images = np.stack(self.images).astype(np.float32)  # 使用32位浮点数
        self.labels = np.concatenate(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 修改后的ResNet的基本块以适应二维数据
class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

# 修改后的ResNet架构
class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet2D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet18的二维版本
def ResNet18_2D():
    return ResNet2D(BasicBlock2D, [2, 2, 2, 2])

# 训练函数
def train(model, dataloader, criterion, optimizer, epochs, device, save_path):
    best_accuracy = 0.0
    model.train()
    model.to(device)
    for epoch in range(epochs):
        print("第"+str(epoch)+"轮训练开始")
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)  # 移动图像到GPU
            labels = labels.to(device)  # 移动标签到GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 评估模型
        current_accuracy = evaluate(model, dataloader, device)
        # 如果当前模型准确率优于历史最佳，则保存模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), save_path)
            print("模型保存成功")

# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # 移动图像到GPU
            labels = labels.to(device)  # 移动标签到GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_list = [f'train_data_0{i}.fits' for i in range(1, 10)] + ['train_data_10.fits']
    
    # 分批加载数据集
    batch_size = 4
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        dataset = SpectraImageDataset(batch_files, transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = ResNet18_2D().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        save_path = "best_model.pth"
        
        # 训练和评估模型
        train(model, train_loader, criterion, optimizer, epochs=5, device=device, save_path=save_path)
        model.load_state_dict(torch.load(save_path))
        evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

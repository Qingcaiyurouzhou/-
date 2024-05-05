import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from astropy.io import fits
import numpy as np

# 数据集类
class SpectraDataset(Dataset):
    def __init__(self, file_list):
        self.fluxes = []
        self.labels = []
        self.objids = []
        for file_name in file_list:
            with fits.open(file_name) as hdulist:
                self.fluxes.append(hdulist[0].data)
                self.labels.append(hdulist[1].data['label'])
                self.objids.append(hdulist[1].data['objid'])
        self.fluxes = np.vstack(self.fluxes)
        self.labels = np.concatenate(self.labels)
        self.objids = np.concatenate(self.objids)

    def __len__(self):
        return len(self.fluxes)

    def __getitem__(self, idx):
        flux = torch.tensor(self.fluxes[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return flux, label

# 修改ResNet的基本块以适应一维数据
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

# 构建ResNet架构
class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
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
        out = nn.AdaptiveAvgPool1d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet18
def ResNet18():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

# 训练函数
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        print("第"+str(epoch)+"轮训练开始")
        for i, (flux, labels) in enumerate(dataloader):
            print("第"+str(i)+"个数据")
            optimizer.zero_grad()
            flux = flux.unsqueeze(1)  # 添加一个通道维度
            outputs = model(flux)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for flux, labels in dataloader:
            flux = flux.unsqueeze(1)  # 添加一个通道维度
            outputs = model(flux)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# 主程序
def main():
    file_list = [f'train_data_0{i}.fits' for i in range(1, 10)] + ['train_data_10.fits']
    dataset = SpectraDataset(file_list)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = ResNet18()
    print("模型加载完成")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, criterion, optimizer, epochs=5)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
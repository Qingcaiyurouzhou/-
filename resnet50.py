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
class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = nn.ReLU(inplace=True)(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out

# 构建ResNet50架构
class ResNet50_1D(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNet50_1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ResNet50
def ResNet50():
    return ResNet50_1D(Bottleneck1D, [3, 4, 6, 3])

# 训练函数
def train(model, dataloader, criterion, optimizer, epochs, device, save_path):
    best_accuracy = 0.0
    model.train()
    model.to(device)
    for epoch in range(epochs):
        print("第"+str(epoch)+"轮训练开始")
        for i, (flux, labels) in enumerate(dataloader):
            print("第"+str(i)+"个数据")
            flux = flux.unsqueeze(1).to(device)  # 添加一个通道维度并移动到GPU
            labels = labels.to(device)  # 移动标签到GPU
            optimizer.zero_grad()
            outputs = model(flux)
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
        for flux, labels in dataloader:
            flux = flux.unsqueeze(1).to(device)  # 添加一个通道维度并移动到GPU
            labels = labels.to(device)  # 移动标签到GPU
            outputs = model(flux)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    file_list = [f'train_data_0{i}.fits' for i in range(1, 10)] + ['train_data_10.fits']
    dataset = SpectraDataset(file_list)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model = ResNet50().to(device)
    print("模型加载完成")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    save_path = "best_model.pth"
    train(model, train_loader, criterion, optimizer, epochs=5, device=device, save_path=save_path)
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(save_path))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

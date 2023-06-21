import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 构建训练集和测试集

input_size = 28     # 图像尺寸28*28
num_classes = 10        # 标签种类数
num_epochs = 3      # 训练周期数
batch_size = 64     # 一个批处理数

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 构建dataloader，可批处理数据集
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 构建卷积网络模块


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,      # 输入通道数，输入大小(1, 28, 28), 通道数为1，灰度图
                out_channels=16,        # 输出通道数(16个卷积核)
                kernel_size=(5, 5),      # 卷积核大小
                stride=(1, 1),       # 步长
                padding=(2, 2)       # 卷积外延尺寸，(input_data_width + 2*padding - kernel_size)/stride + 1 = out_data_width
            ),      # 输出大小(16, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)       # 池化
        )       # 输出大小(16, 14, 14)
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, (5, 5), (1, 1), (2, 2)),       # 输出大小(32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),       # 输出大小(32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)       # 输出大小(32, 7, 7)
        )
        self.conv_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),       # 输出大小(64, 7, 7)
            torch.nn.ReLU()
        )
        self.out = torch.nn.Linear(64 * 7 * 7, 10)      # 全连接，输出类别为10

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), -1)       # 数据flatten操作，将数据构造为两维度（批处理数量和数据）, 构造数据为(batch_size, 32*7*7)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    """
    计算准确率
    返回数据为(正确数，总数)
    相除为准确率
    :param predictions:
    :param labels:
    :return:  正确数，总数
    """
    pred = torch.max(input=predictions.data, dim=1)[1]      # 返回预测最大值得种类索引
    right_num = pred.eq(labels.data.view_as(pred)).sum()        # 预测正确的数量
    return right_num, len(labels)


# 训练网络模型

cnn = CNN()
# 损失函数
cost = torch.nn.CrossEntropyLoss()
# 参数优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # 当前epoch结果
    train_right_num_and_labels_len = list()

    for batch_index, (train_data, train_target) in enumerate(train_dataloader):
        cnn.train()     # 切换模型为训练模式
        train_output = cnn(train_data)      # 训练数据
        loss = cost(train_output, train_target)     # 计算损失
        optimizer.zero_grad()       # 优化器清空梯度记录
        loss.backward()     # 根据损失反向传播
        optimizer.step()        # 优化器迭代参数
        # 返回并收集该批量预测正确数和该批量总数
        train_right_num, train_labels_len = accuracy(train_output, train_target)
        train_right_num_and_labels_len.append((train_right_num, train_labels_len))

        if batch_index % 100 == 0:
            # 每100批进行一次测试集效果评估且打印
            cnn.eval()      # 切换模型为评估模式
            test_right_num_and_labels_len = list()
            for test_data, test_target in test_dataloader:
                test_output = cnn(test_data)        # 预测数据
                # 返回并收集该批量预测正确数和该批量总数
                test_right_num, test_labels_len = accuracy(test_output, test_target)
                test_right_num_and_labels_len.append((test_right_num, test_labels_len))

            # 当前已训练过的训练集预测过的所有正确数和所有总数
            train_right_sum_and_labels_len_sum = (
                sum(i[0] for i in train_right_num_and_labels_len),
                sum(i[1] for i in train_right_num_and_labels_len)
            )
            # 测试集所有预测正确数和所有总数
            test_right_sum_and_labels_len_sum = (
                sum(i[0] for i in test_right_num_and_labels_len),
                sum(i[1] for i in test_right_num_and_labels_len)
            )
            # 计算当前训练集正确率和当前模型测试集正确率
            train_right_rate = train_right_sum_and_labels_len_sum[0] / train_right_sum_and_labels_len_sum[1]
            test_right_rate = test_right_sum_and_labels_len_sum[0] / test_right_sum_and_labels_len_sum[1]

            print(f'当前epoch:{epoch}，进度[{batch_index}/{len(train_dataloader)} '
                  f'{100*batch_index/len(train_dataloader): .2f}%]\t'
                  f'损失:{loss.data: .6f}\t训练集准确率: {100*train_right_rate: .6f}%\t'
                  f'测试集准确率: {100*test_right_rate: .6f}%')



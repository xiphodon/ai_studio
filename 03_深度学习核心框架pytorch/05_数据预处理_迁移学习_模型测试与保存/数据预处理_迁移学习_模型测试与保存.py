import copy
import json
import time
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

import torch.cuda
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

data_dir = Path('./data/flower_data')
train_dir = data_dir / 'train'
valid_dir = data_dir / 'valid'
cat_to_name_file_path = Path('./data/cat_to_name.json')

# 数据预处理
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize(size=(96, 96)),  # 尺寸统一
            transforms.RandomRotation(degrees=45),  # 随机旋转-45度~+45度
            transforms.CenterCrop(size=(64, 64)),  # 中心裁剪出64*64像素图片
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，0.5概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，0.5概率
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度，对比度，饱和度，色相
            transforms.RandomGrayscale(p=0.025),  # 随机灰度处理，0.025概率，三通道灰度即R=G=B，各个通道值一致
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 数据标准化，大数据集经验值为佳
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
}

batch_size = 128
image_dataset = {
    k: datasets.ImageFolder(
        root=(data_dir / k).as_posix(),
        transform=data_transforms[k]
    ) for k in ['train', 'valid']
}
dataloader = {
    k: DataLoader(
        dataset=image_dataset[k],
        batch_size=batch_size,
        shuffle=True
    ) for k in ['train', 'valid']
}
dataset_sizes = {k: len(image_dataset[k]) for k in ['train', 'valid']}
class_names = image_dataset['train'].classes
pprint(image_dataset)
print(dataset_sizes)

# 标签对应实际名字
with open(file=cat_to_name_file_path.as_posix(), mode='r') as fp:
    cat_to_name = json.load(fp)

print(f'类别共计：{len(cat_to_name)}', cat_to_name)

# 加载经典模型和其权重，迁移学习
model_name = 'resnet'
# 特征提取为True，为使用经典模型特征，不做训练与更新
feature_extract = True

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('### training on GPU ###')
else:
    print('### training on CPU ###')
device = torch.device('cuda:0' if train_on_gpu else 'cpu')


def set_parameter_requires_grad(model: torch.nn.Module, feature_extracting: bool):
    """
    根据是否使用经典模型权重，来设置模型各个参数是否更新其梯度值
    :param model:
    :param feature_extracting:
    :return:
    """
    if feature_extracting:
        # 使用经典模型权重特征
        for param in model.parameters():
            param.requires_grad = False


# 选用resnet 18层模型
# model_resnet = models.resnet18()
# print(model_resnet)


# 修改模型输出层为自己需要的
def initialize_model(feature_extracting: bool, use_pretrained=True):
    """
    初始化模型
    :param feature_extracting:
    :param use_pretrained:
    :return:
    """
    resnet_18 = models.resnet18(pretrained=use_pretrained)  # 使用预训练权重
    set_parameter_requires_grad(resnet_18, feature_extracting=feature_extracting)  # 冻结所有层，禁止梯度更新

    num_fc_in_features = resnet_18.fc.in_features  # 获取原最后一层fc全连接层的输入特征
    resnet_18.fc = torch.nn.Linear(in_features=num_fc_in_features, out_features=102)  # 构造新的全连接层fc，并接入网络

    return resnet_18


resnet_18 = initialize_model(feature_extracting=feature_extract)  # 初始化模型
resnet_18 = resnet_18.to(device=device)  # gpu训练

# 模型保存名字
model_file_name = 'resnet18_1.pth'

print('params to learn: \t')
params_to_update = list()  # 需要计算梯度的参数
# 仅获取未冻结层需要计算梯度的参数
for name, param in resnet_18.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print(name, param)

print(resnet_18)

# 参数优化器设置
optimizer = torch.optim.Adam(params_to_update, lr=0.01)
# 学习率衰减调度器，epoch每经过step_size步，学习率衰减为原来的gamma倍
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
# 损失函数
cost = torch.nn.CrossEntropyLoss()


# 训练模块
def train_model(model: torch.nn.Module, dataloader, cost, optimizer: torch.optim.Optimizer, num_epochs=25, filename='best.pt'):
    """
    训练模型
    :param model:
    :param dataloader:
    :param cost:
    :param optimizer:
    :param num_epochs:
    :param filename:
    :return:
    """
    # 记录时间
    time_1 = time.time()
    # 最佳准确率
    best_acc = 0
    # 模型运行在gpu or cpu
    model.to(device=device)

    # 训练过程中指标集合
    train_acc_history = list()
    valid_acc_history = list()
    train_loss = list()
    valid_loss = list()
    # 收集学习率
    lr_list = list()

    # 备份当前模型状态（权重和偏移）
    best_model_weights = None

    for epoch in range(num_epochs):
        print(f'epoch: {epoch}/{num_epochs - 1}')
        print('-' * 20)

        for k in ['train', 'valid']:
            if k == 'train':
                model.train()   # 训练模式
            else:
                model.eval()    # 验证模式

            running_loss = 0
            running_corrects = 0

            for input_bat, labels_bat in dataloader[k]:
                # 将数据放入指定gpu or cpu
                input_bat = input_bat.to(device)
                labels_bat = labels_bat.to(device)

                # 缓存梯度清零
                optimizer.zero_grad()
                # 模型计算数据
                output_bat = model(input_bat)
                # 计算损失
                loss_bat = cost(output_bat, labels_bat)
                # print(f'loss_bat: {loss_bat}')
                # max_value_index 即为最大概率对应的类别索引
                max_value, max_value_index = torch.max(input=output_bat, dim=1)

                if k == 'train':
                    # 训练阶段需要反向传播计算梯度并更新权重数据
                    loss_bat.backward()
                    optimizer.step()

                # 累计总损失， input_bat.size(0)，第0维度即为该批数据的数量
                running_loss += loss_bat.item() * input_bat.size(0)
                # 累计预测正确数
                running_corrects += torch.sum(max_value_index == labels_bat.data)

            # 总平均损失
            epoch_loss = running_loss / len(dataloader[k].dataset)
            # 总准确率
            epoch_acc = running_corrects / len(dataloader[k].dataset)

            # 记录训练和验证阶段各指标
            if k == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                valid_acc_history.append(epoch_acc)
                valid_loss.append(epoch_loss)

            time_used_epoch = time.time() - time_1
            # 打印使用时间和指标
            print(f'time: {time_used_epoch: .1f}s')
            print(f'{k} loss: {epoch_loss: .4f}, acc: {epoch_acc: .4f}')

            # 记录最佳模型参数
            if k == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = OrderedDict(copy.deepcopy(model.state_dict()))
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, filename)

        print(f'optimizer learning rate: {optimizer.param_groups[0]["lr"]}')
        lr_list.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()     # 学习率衰减调度器走一步
        print('=' * 30)

    time_used = time.time() - time_1
    print(f'time: {time_used: .1f}s')
    print(f'best acc: {best_acc: .4f}')

    # 模型加载最佳权重参数
    model.load_state_dict(best_model_weights)
    return model, train_acc_history, valid_acc_history, train_loss, valid_loss, lr_list


# # 开始训练，当前仅训练输出层（全连接层）
# model, train_acc_history, valid_acc_history, train_loss, valid_loss, lr_list = train_model(
#     resnet_18, dataloader, cost, optimizer
# )


########## 训练所有参数
model = resnet_18

# 得到比随机数好的全连接层权重参数后，解冻模型所有层，训练所有权重
for param in model.parameters():
    param.requires_grad = True

# 训练所有权重参数，调小学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)

# 读取模型文件，并取出对应参数
best_state = torch.load('best.pt')
best_acc = best_state['best_acc']
best_state_dict = best_state['state_dict']

# 模型加载权重参数
model.load_state_dict(best_state_dict)

# # 开始训练，训练所有参数
# model, train_acc_history, valid_acc_history, train_loss, valid_loss, lr_list = train_model(
#     resnet_18, dataloader, cost, optimizer
# )

#############  使用训练好的模型和参数，可视化数据查看预测
# 加载一小批量数据
test_dataiter = iter(dataloader['valid'])
images, labels = next(test_dataiter)
# 切换为评估模式
model.eval()

# if train_on_gpu:
#     output = model(images.cuda())
# else:
#     output = model(images)

images = images.to(device)
output = model(images)

print(output.shape)

# 最大数值， 最大数值对应的类别索引
max_value, max_value_index = torch.max(input=output, dim=1, keepdim=False)

if train_on_gpu:
    # gpu张量数据需要先copy到cpu才能转换为numpy类型
    max_value_index_np = max_value_index.cpu().numpy()
else:
    max_value_index_np = max_value_index.numpy()

print(max_value_index_np)


def image_transform(img: Tensor):
    """
    图片数据展示
    :param img:
    :return:
    """
    # 张量数据移动至cpu转为numpy，并不需要保持维度
    img = img.to('cpu').numpy().squeeze()
    # 将色彩通道从第一个维度转变到第三个维度
    img = img.transpose(1, 2, 0)
    # 将标准化数据还原为原始数据    x` = (x-u)/s  =>  x = x`*s + u
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # 修剪数据，使元素数值保持在0~1
    img = img.clip(0, 1)
    return img


fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for i in range(rows * columns):
    ax = fig.add_subplot(rows, columns, i + 1)
    # 从小批量数据中取一张图，并压缩维度，第一个维度为图片数量，无需保持该维度
    img = images[i].squeeze()
    plt.imshow(image_transform(img))

    # 预测类别对应文字
    prediction_text = cat_to_name[str(max_value_index_np[i])]
    # 标签类别对应文字
    label_text = cat_to_name[str(labels[i].item())]

    # 显示对应种类的名称
    ax.set_title(f'{prediction_text}({label_text})', color=('green' if prediction_text == label_text else 'red'))

plt.show()

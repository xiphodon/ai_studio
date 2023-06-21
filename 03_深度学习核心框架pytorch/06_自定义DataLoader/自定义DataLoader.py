from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 使用 ../05_数据预处理_迁移学习_模型测试与保存/data 中数据来制作自定义DataLoader
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data_dir_path = Path(r'../05_数据预处理_迁移学习_模型测试与保存/data')
flower_data_dir_path = data_dir_path / 'flower_data'
cat_to_name_file_path = data_dir_path / 'cat_to_name.json'

train_dir_path = flower_data_dir_path / 'train'
valid_dir_path = flower_data_dir_path / 'valid'

# # train data
# train_img_path_list = list()
# train_img_label_list = list()
# for label_dir_path in train_dir_path.iterdir():
#     label = label_dir_path.name
#     for img_path in label_dir_path.iterdir():
#         train_img_path_list.append(img_path.as_posix())
#         train_img_label_list.append(np.array(label, dtype=np.int64))
# print(train_img_path_list)
# print(train_img_label_list)


# 构造自定义Dataset类
class FlowerDataset(Dataset):
    """
    花朵数据集
    """
    def __init__(self, train_or_valid: str = 'train', transform=None):
        self.train_or_valid = train_or_valid
        if train_or_valid == 'train':
            self.train_or_valid_dir_path = train_dir_path
        elif train_or_valid == 'valid':
            self.train_or_valid_dir_path = valid_dir_path
        else:
            raise ValueError('train_or_valid value must in ("train" or "valid")')
        self.img_path_list, self.img_label_list = self.get_img_path_list_and_label_list()
        self.transform = transform

    def get_img_path_list_and_label_list(self):
        """
        获取图片路径列表和标签列表
        :return:
        """
        img_path_list = list()
        img_label_list = list()
        for label_dir_path in self.train_or_valid_dir_path.iterdir():
            label = label_dir_path.name
            for img_path in label_dir_path.iterdir():
                img_path_list.append(img_path.as_posix())
                img_label_list.append(np.array(label, dtype=np.int64))
        return img_path_list, img_label_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(fp=self.img_path_list[index])
        label = self.img_label_list[index]
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(label)
        return img, label


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


train_dataset = FlowerDataset(train_or_valid='train', transform=data_transforms['train'])
valid_dataset = FlowerDataset(train_or_valid='valid', transform=data_transforms['valid'])

# 通过实例化的自定义dataset构造DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)

print(len(train_dataset))
print(len(valid_dataset))

# 取出一批数据测试
img_bat, label_bat = next(iter(train_dataloader))
img_item: Tensor = img_bat[0].squeeze()      # 取出第一条数据，并无需保持第一个维度，压缩掉批数量维度
img_item = img_item.permute(1, 2, 0).numpy()      # 将tensor中通道维度转换为第三维度

img_item *= [0.229, 0.224, 0.225]
img_item += [0.485, 0.456, 0.406]

plt.imshow(img_item)
plt.show()
print(f'img_item label: {label_bat[0].numpy()}')



from pathlib import Path
import pickle
import gzip
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


# http://deeplearning.net/data/mnist/mnist.pkl.gz
data_path = Path('data') / 'mnist' / 'mnist.pkl.gz'

with gzip.open(data_path.as_posix(), 'rb') as fp:
    (x_train, y_train), (x_valid, y_valid), _ = pickle.load(fp, encoding='latin-1')

# (50000, 784) 50000条数据，每条数据784个特征，即28*28像素图片
print(x_train.shape)
# plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
# plt.show()

# 转换为tensor类型
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

x_train_size, x_train_feature_size = x_train.shape
# print(x_train[0], y_train[0])
# print(y_train.min())
# print(y_train.max())


class MnistNN(torch.nn.Module):
    """
    手写数字识别神经网络
    """
    def __init__(self):
        super().__init__()
        self.hidden_1 = torch.nn.Linear(784, 128)
        self.hidden_2 = torch.nn.Linear(128, 256)
        self.out = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = f.relu(self.hidden_1(x))
        x = f.relu(self.hidden_2(x))
        x = self.out(x)
        return x


mnist_nn = MnistNN()
print(mnist_nn)

# for name, parameter in mnist_nn.named_parameters():
#     print(name, parameter, parameter.size(), sep='\n')
#     print('-' * 20)

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)

loss_func = f.cross_entropy


def get_data(train_ds, valid_ds):
    return (
        DataLoader(train_ds, batch_size=64, shuffle=True),
        DataLoader(valid_ds, batch_size=128)
    )


def loss_batch(model: torch.nn.Module, loss_func, x_bat, y_bat, opt=None):
    """
    批量计算损失
    :param model:
    :param loss_func:
    :param x_bat:
    :param y_bat:
    :param opt:
    :return:
    """
    loss = loss_func(model(x_bat), y_bat)

    if opt:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x_bat)


def fit(steps, model: torch.nn.Module, loss_func, opt, train_dl, valid_dl):
    """
    训练模型
    :param steps:
    :param model:
    :param loss_func:
    :param opt:
    :param train_dl:
    :param valid_dl:
    :return:
    """
    for step in range(steps):
        model.train()
        for x_bat, y_bat in train_dl:
            loss_batch(model, loss_func, x_bat, y_bat, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, x_bat, y_bat) for x_bat, y_bat in valid_dl]
            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f'当前step:{step}, 验证集损失{val_loss}')


def get_model():
    """
    获取模型
    :return:
    """
    model = MnistNN()
    return model, optim.Adam(model.parameters(), lr=0.001)


train_dl, valid_dl = get_data(train_dataset, valid_dataset)
model, opt = get_model()
fit(steps=10, model=model, loss_func=loss_func, opt=opt, train_dl=train_dl, valid_dl=valid_dl)
print(torch.max(model(x_valid[:10]), dim=1), y_valid[:10])

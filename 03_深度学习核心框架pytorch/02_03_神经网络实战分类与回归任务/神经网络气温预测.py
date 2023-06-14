import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.optim as optim
import warnings

data = pd.read_csv('./data/temps.csv')
# print(data.head())

# year,moth,day,week分别表示的具体的时间
# temp_2：前天的最高温度值
# temp_1：昨天的最高温度值
# average：在历史中，每年这一天的平均最高温度值
# actual：当天的真实最高温度
# friend：朋友猜测的可能值

# (348, 9)
print(data.shape)

years = data['year']
months = data['month']
days = data['day']

date_str_list = [f'{year}-{month}-{day}' for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date_item, '%Y-%m-%d') for date_item in date_str_list]

# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=90)

# 标签值
ax1.plot(dates, data['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates, data['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates, data['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')

# 朋友猜测
ax4.plot(dates, data['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
plt.show()

# 将分类数据转换为独热编码
data = pd.get_dummies(data)
# print(data.head())

labels = np.array(data['actual'])
del data['actual']
data_name = list(data.columns)

data = np.array(data)
print(data.shape)

# 数据标准化
data = preprocessing.StandardScaler().fit_transform(data)

# # 构建网络模型
# x = torch.tensor(data, dtype=torch.float)
# y = torch.tensor(labels, dtype=torch.float)
#
# w_1 = torch.randn(size=(14, 128), dtype=torch.float, requires_grad=True)
# b_1 = torch.randn(size=(128, ), dtype=torch.float, requires_grad=True)
# w_2 = torch.randn(size=(128, 1), dtype=torch.float, requires_grad=True)
# b_2 = torch.randn(size=(1,), dtype=torch.float, requires_grad=True)
#
# learning_rate = 0.001
# losses = list()
#
# for i in range(1, 1001):
#     # 计算隐层
#     hidden = torch.relu(x.mm(w_1) + b_1)
#     # 输出
#     out = hidden.mm(w_2) + b_2
#
#     # 计算损失
#     loss = torch.mean((out - y)**2)
#     losses.append(loss.data.numpy())
#
#     if i % 100 == 0:
#         print(f'loss: {loss}')
#
#     # 反向传播
#     loss.backward()
#
#     # 更新参数
#     w_1.data.add_(-learning_rate*w_1.grad.data)
#     b_1.data.add_(-learning_rate*b_1.grad.data)
#     w_2.data.add_(-learning_rate*w_2.grad.data)
#     b_2.data.add_(-learning_rate*b_2.grad.data)
#
#     # 清空梯度记录
#     w_1.grad.data.zero_()
#     b_1.grad.data.zero_()
#     w_2.grad.data.zero_()
#     b_2.grad.data.zero_()


# 构建网络模型方式2

feature_size = data.shape[1]
hidden_1_size = 128
hidden_2_size = 64
output_size = 1
batch_size = 16

# 构造网络
my_nn = torch.nn.Sequential(
    torch.nn.Linear(feature_size, hidden_1_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_1_size, hidden_2_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_2_size, output_size)
)
# 损失函数
loss = torch.nn.MSELoss(reduction='mean')
# 参数优化器
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = list()
for i in range(1, 1001):
    batch_loss = list()
    # 小批量梯度下降
    for bat_start_index in range(0, len(data), batch_size):
        # 一批数据尾索引
        bat_end_index = min(bat_start_index + batch_size, len(data))
        x_bat = torch.tensor(data[bat_start_index: bat_end_index], dtype=torch.float, requires_grad=True)
        y_bat = torch.tensor(labels[bat_start_index: bat_end_index], dtype=torch.float, requires_grad=True)
        prediction = my_nn(x_bat)

        # 计算损失并记录
        # loss_output = loss(torch.reshape(prediction, shape=(-1,)), y_bat)
        loss_output = loss(prediction, torch.reshape(y_bat, shape=(-1, 1)))
        batch_loss.append(loss_output.data.numpy())

        # 优化器梯度计算清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss_output.backward(retain_graph=True)
        # 优化器更新参数
        optimizer.step()

    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(f'step: {i}, loss: {np.mean(batch_loss)}')

# 查看训练结果
x = torch.tensor(data, dtype=torch.float)
predict = my_nn(x).data.numpy()

true_data_df = pd.DataFrame(data={'date': dates, 'actual': labels})
prediction_data_df = pd.DataFrame(data={'date': dates, 'predictions': predict.reshape(-1)})

plt.plot(true_data_df['date'], true_data_df['actual'], 'b--', label='actual')
plt.plot(prediction_data_df['date'], prediction_data_df['predictions'], 'ro', label='prediction')
plt.xticks(rotation='90')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')

plt.show()

import torch
import torch.nn as nn
import numpy as np
import time

class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(vocab_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 加载数据，具体实现取决于你的数据结构
data = np.load('/Users/cglin/Desktop/data/3/tang.npz')
# time.sleep(100)
poems = data['data']
# 需要进行标记化和整数索引转换

# 创建模型
vocab_size = 2000  # 字典的大小
hidden_size = 128  # 你希望的隐藏状态的大小
output_size = vocab_size  # 输出的大小通常与词汇量相同
model = RNNModel(vocab_size, hidden_size, output_size)

# 读取训练过的模型
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 开始预测
start_token = ...  # 需要转换为整数索引
hidden = model.initHidden()
output_poem = [start_token]
for i in range(max_length):
    output, hidden = model(output_poem[i], hidden)
    topv, topi = output.topk(1)
    output_poem.append(topi.item())
    if topi.item() == end_token:
        break

# 将 output_poem 转回字符或单词

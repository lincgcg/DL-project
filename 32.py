import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
# from keras.utils import np_utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# 读取npz文件
data = np.load('/Users/cglin/Desktop/data/3/tang.npz')
poems = data['data']



poems = ["".join(map(str, poem)) for poem in poems]

dataloader = DataLoader(poems, batch_size=32, shuffle=True)


# 将字符转化为数字
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(list(set(''.join(poems))))

# 对y进行one-hot编码
# onehot_encoder = np_utils.to_categorical(integer_encoded)
onehot_encoder = np.eye( len(label_encoder.classes_))[integer_encoded]


# 构建一个字符到数字和数字到字符的映射
char_to_int = dict((c, i) for i, c in enumerate(label_encoder.classes_))
int_to_char = dict((i, c) for i, c in enumerate(label_encoder.classes_))

# 定义序列长度
seq_length = 100

# 切分为训练数据和标签
dataX = []
dataY = []
for poem in poems:
    for i in range(0, len(poem) - seq_length, 1):
        seq_in = poem[i:i + seq_length]
        seq_out = poem[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# 参数
n_epochs = 5000
print_every = 200
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

n_characters = len(label_encoder.classes_)

# 初始化模型，损失函数和优化器
model = RNN(n_characters, hidden_size, n_characters, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 开始训练
start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        loss = train(inputs)
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (start), epoch, epoch / n_epochs * 100, loss)
            print(evaluate('Wh', 100), '\n')

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = model.init_hidden(1)
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

evaluate()
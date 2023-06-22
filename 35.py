import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import time


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))

    def load(self, path):

        self.load_state_dict(torch.load(path))

    def save(self, name=None):

        if name is None:
            prepath = '/Users/cglin/Desktop/output/3/models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M.pth')
        torch.save(self.state_dict(), name)
        print("保存的模型路径为：", name)
        return name

class PoetryModel(BasicModule):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.modelName = 'PoetryModel'
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, vocab_size)
        )

    def forward(self, input, hidden = None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output.view(seq_len * batch_size, -1))
        return output, hidden

def poetryData(filename, batch_size):

    dataset = np.load(filename, allow_pickle=True)
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()

    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, ix2word, word2ix

class Accumulator():

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def train(model,filename, batch_size, lr, epochs, device, pre_model_path=None):

    if pre_model_path:
        model.load(pre_model_path)
    model.to(device)

    dataloader, ix2word, word2ix = poetryData(filename, batch_size)

    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    metric = Accumulator(2)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)

            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * data.shape[0], data.shape[0])
            train_loss = metric[0] / metric[1]
            if i % 15 == 0:
                print('epoch: {:d}, batch: {:d}, Train Loss: {:.4f}'.format(epoch, i, train_loss))
        scheduler.step(train_loss)

    model.save()

def generate(model, filename, device, start_words, max_gen_len, prefix_words=None):

    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = list(start_words)
    start_word_len = len(start_words)

    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break

    return results

def generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic):

    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = []
    start_word_len = len(start_words_acrostic)
    index = 0
    pre_word = '<START>'

    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        if (pre_word in {u'。', u'！', '<START>'}):
            if index == start_word_len:
                break
            else:
                w = start_words_acrostic[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    return results

if __name__ == "__main__":

    filename = r'/Users/cglin/Desktop/data/3/tang.npz'
    batch_size = 64
    lr = 0.001
    epochs = 10
    vocab_size = 8293
    embedding_dim = 128
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    #模型训练
    # model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    # # visdir = time.strftime( 'assets/visualize/' + model.modelName + '_%m%d_%H_%M')
    # # trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    # train(model, filename, batch_size, lr, epochs, device, pre_model_path=None)
    

    
    # #给定开头生成诗句
    # model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    # model.load('/Users/cglin/Desktop/output/3/models/PoetryModel_0619_10_21.pth')
    # start_words = '纸上得来终觉浅'
    # max_gen_len = 128
    # prefix_words = None
    # poetry = ''
    # result = generate(model, filename, device, start_words, max_gen_len, prefix_words)
    # for word in result:
    #     poetry += word
    #     if word == '。' or word == '!':
    #         poetry += '\n'
    # print(poetry)
    

    #生成藏头诗
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    model.load('/Users/cglin/Desktop/output/3/models/PoetryModel_0619_10_21.pth')
    start_words_acrostic = '深度学习'
    max_gen_len_acrostic = 128
    prefix_words_acrostic = None
    poetry = ''
    result = generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic)
    for word in result:
        poetry += word
        if word == '。' or word == '!':
            poetry += '\n'
    print(poetry)

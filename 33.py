import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
# from keras.utils import np_utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

def prepareData():
    datas = np.load("/Users/cglin/Desktop/data/3/tang.npz")

    data = datas['data']

    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=2)
    return dataloader, ix2word, word2ix

class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):

        embeds = self.embeddings(input) 
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
            output, hidden = self.lstm(embeds, (h_0, c_0))
            output = self.linear(output)
            output = output.reshape(batch_size * seq_len, -1)
        
        return output, hidden

def train(dataloader, word2ix):
    model = PoetryModel(len(word2ix), embedding_dim=Config.embedding_dim, hidden_dim=Config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

def generate(model, start_words, ix2word, word2ix):

    results = list(start_words)
    start_words_len = len(start_words) # 第一个词语是<START> 
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    hidden = None
    model.eval()
    with torch.no_grad():

        for i in range(Config.max_gen_len):

            output, hidden = model(input, hidden)

            # 如果在给定的句首中，input 为句首中的下一个字

            if i < start_words_len:

                w = results[i]

                input = input.data.new([word2ix[w]]).view(1, 1) # 否则将 output 作为下一个 input 进行 
            else:

                top_index = output.data[0].topk(1)[1][0].item()

                w = ix2word[top_index]

                results.append(w)
                
                input = input.data.new([top_index]).view(1, 1)

            if w == '<EOP>':

                del results[-1]
                break 
        return results

model = PoetryModel(2000,128,256)

dataloader, ix2word, word2ix = prepareData()

train(dataloader, word2ix)

start_words = ['<START>', "窗", "前", "明"]

generate(model, start_words, ix2word, word2ix)

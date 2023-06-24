import torch
from torch import nn
import torch.nn.functional as F
import time
import torch.optim as optim
import torch.utils.data as Data
from zhconv import convert
import gensim
import numpy as np
from torch.utils.data import Dataset
import wandb
import argparse
import os

class BasicModule(nn.Module):
    """
    封装nn.Module，提供load模型和save模型接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))

    def load(self, path):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存训练的模型到指定路径
        '''
        if name is None:
            prepath = '/Users/cglin/Desktop/output/4/models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        print("保存的模型路径为：", name)
        return name

class TextCNN(BasicModule):
    def __init__(self, vocab_size, embedding_dim, filters_num, filter_size, pre_weight):
        super(TextCNN, self).__init__()
        self.modelName = 'TextCNN'
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = False
        if pre_weight is not None:
            self.embeddings = self.embeddings.from_pretrained(pre_weight)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filters_num, (size, embedding_dim)) for size in filter_size])
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(filters_num * len(filter_size), 2)
        self.fc = nn.Sequential(
            nn.Linear(filters_num * len(filter_size), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        # self.fc1 = nn.Linear(filters_num * len(filter_size), 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 2)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(filters_num * len(filter_size), 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)  # Dropout after fc1
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)  # Dropout after fc2
        # )
        # self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        '''
        x的size为(batch_size, max_len)
        '''
        x = self.embeddings(x)  #(batch_size, max_len, embedding_dim)
        x = x.unsqueeze(1)      #(batch_size, 1, max_len, embedding_dim)
        x = torch.tensor(x, dtype=torch.float32)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # out = self.fc3(x)
        return out

class LSTMModel(BasicModule):
    def __init__(self, embedding_dim, hidden_dim, pre_weight):
        super(LSTMModel, self).__init__()
        self.modelName = 'LSTMModel'
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        self.embeddings.weight.requires_grad = False
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3, batch_first=True, dropout=0.5)
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.hidden_dim, 2),
        #     nn.Sigmoid()
        # )
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, input, hidden = None):
        '''
        input的size为(batch_size, max_len)
        '''
        batch_size, max_len = input.size()
        embeds = self.embeddings(input)
        embeds = torch.tensor(embeds, dtype=torch.float32)
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output)
        #取最后一个时间步的输出
        last_outputs = self.get_last_output(output, batch_size, max_len)
        return last_outputs, hidden

    def get_last_output(self, output, batch_size, max_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(batch_size):
            last_outputs[i] = output[i][max_len - 1]
        last_outputs = last_outputs.to(output.device)
        return last_outputs

def build_word2id(trainpath, validatepath, testpath):
    """
    :param file: word2id保存地址
    :return: 返回id2word、word2id
    """
    word2id = {'_PAD_': 0}
    id2word = {0: '_PAD_'}
    paths = [trainpath, validatepath, testpath]
    #print(path)
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                line = convert(line, 'zh-cn')
                words = line.strip().split()
                for word in words[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    for key, val in word2id.items():
        id2word[val] = key
    return word2id, id2word

def build_word2vec(file, word2id, save_to_path=None):
    """
    :param file: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    word2vecs = torch.from_numpy(word_vecs)
    return word2vecs

class CommentDataset(Dataset):
    def __init__(self, file, word2id, id2word):
        self.file = file
        self.word2id = word2id
        self.id2word = id2word
        self.datas, self.labels = self.getboth()

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def getboth(self):
        datas, labels = [], []
        with open(self.file, encoding='utf-8') as f:
            for line in f.readlines():
                #取每行的label
                label = torch.tensor(int(line[0]), dtype=torch.int64)
                labels.append(label)
                #取每行的word
                line = convert(line, 'zh-cn')
                line_words = line.strip().split()[1:-1]
                indexs = []
                for word in line_words:
                    try:
                        index = self.word2id[word]
                    except BaseException:
                        index = 0
                    indexs.append(index)
                datas.append(indexs)
            return datas, labels

def mycollate_fn(data):
    #step1: 分离data、label
    data.sort(key=lambda x: len(x[0]), reverse=True)
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])

    #step2: 大于75截断、小于75补0
    padded_datas = []
    for data in input_data:
        if len(data) >= 75:
            padded_data = data[:75]
        else:
            padded_data = data
            while (len(padded_data) < 75):
                padded_data.append(0)
        padded_datas.append(padded_data)

    #step3: label、data转为tensor
    label_data = torch.tensor(label_data)
    padded_datas = torch.tensor(padded_datas, dtype=torch.int64)
    return padded_datas, label_data

class Accumulator():
    """
    构建n列变量，每列累加，便于计算准确率与损失
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def sameNumber(y_hat, y):
    """
    返回预测值与真实值相等的个数
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()

def train(model, batch_size, lr, epochs, device, trainloader, validateloader):
    """
    描述：训练模型并计算损失、准确率
    """
    #step1: 将模型设置到device上
    model.to(device)
    model.train()

    #step2: 定义目标函数与优化器，规定学习率衰减规则
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

    #step3: 训练模型计算损失、准确率并在每个epoch进行验证
    metric = Accumulator(3)
    step = 0
    best_val = 0
    path = "/Users/cglin/Desktop/output/4/models/save.bin"
    for epoch in range(epochs):
        # if os.path.exists(path):
        #     print("loading model")
        #     model.load_state_dict(torch.load(path))
        #     model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            if model.modelName == 'LSTMModel':
                outputs, hidden = model(inputs)
            elif model.modelName == 'TextCNN':
                outputs = model(inputs)
            #print(outputs, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            with torch.no_grad():
                metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # wandb.log('Batch Loss', train_loss)
            if i % 50 == 0:
                print('epoch: {:d}, batch: {:d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, i, train_loss, train_acc))
        scheduler.step(train_loss)
        # wandb.log('train_Epoch Loss', train_loss)
        # wandb.log('train_Accuracy', train_acc)

        # 验证，计算验证损失和准确率
        validate_loss, validate_acc = validate(model, validateloader, criterion)
        print('epoch: {:d}, Validate Loss: {:.4f}, Validate Accuracy: {:.4f}'.format(epoch, validate_loss, validate_acc))
        print("=====valid=====")
        print("epoch num = ")
        print(epoch)
        print("valid_Epoch Loss")
        print(validate_loss)
        print("valid_accuracy")
        print(validate_acc)
        print("==============")
        # wandb.log('valid_Epoch Loss', validate_loss)
        # wandb.log('valid_ccuracy', validate_acc)

        # # 当验证准确度达到83%以上时保存模型
        # if train_acc >0.83 and validate_acc > 0.83 :
        #     model.save()
        #     print("该模型是在第{}个epoch取得80%以上的验证准确率, 准确率为：{:.4f}".format(epoch, validate_acc))
        if validate_acc > best_val:
            print("validate_acc")
            print(validate_acc)
            model.save(path)
            best_val = validate_acc
        else :
            print("Best validate acc")
            print(best_val)
        print("==============")


    #step5: 返回模型用于测试
    return model

def validate(model, validateloader, criterion):
    """
    描述：验证模型的准确率并计算损失
    """
    model.eval()
    metric = Accumulator(3)
    for i, data in enumerate(validateloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if model.modelName == 'LSTMModel':
            outputs, hidden = model(inputs)
        elif model.modelName == 'TextCNN':
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
        validate_loss = metric[0] / metric[2]
        validate_acc = metric[1] / metric[2]
    model.train()
    return validate_loss, validate_acc

def test(model, device, testloader):
    """
    描述：测试模型的准确率并计算损失
    """
    #step1: 定义目标函数
    criterion = nn.CrossEntropyLoss()

    #step2: 测试，计算测试损失和准确率
    metric = Accumulator(3)
    model.to(device)
    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if model.modelName == 'LSTMModel':
            outputs, hidden = model(inputs)
        elif model.modelName == 'TextCNN':
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
        test_loss = metric[0] / metric[2]
        test_acc = metric[1] / metric[2]
        # wandb.log('test_Loss', test_loss)
        # wandb.log('test_Accuracy', test_acc)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))
    return test_acc

if __name__ == "__main__":
    
    # 初始化wanda
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="DL-project-4",
    #     name = "Text-CNN",
    #     # track hyperparameters and run metadata
    #     config={
    #     "Batch size": 64,
    #     "model": "CNN"
    #     }
    # )
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--lr", type=float, default=0.01,
                            help="Learning rate")
    args = parser.parse_args()
    
    # 模型训练
    batch_size = 128
    # batch_size = 256
    lr = args.lr
    epochs = 5
    embedding_dim = 50
    vocab_size = 57080
    hidden_dim = 1024
    # hidden_dim = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainpath = r'/Users/cglin/Desktop/data/4/train.txt'
    validatepath = r'/Users/cglin/Desktop/data/4/validation.txt'
    testpath = r'/Users/cglin/Desktop/data/4/test.txt'
    word2vec_pretrained = r'/Users/cglin/Desktop/output/4/wiki_word2vec_50.bin'

    #step1: 生成训练、验证、测试数据
    word2id, id2word = build_word2id(trainpath, validatepath, testpath)
    print("length of word2id:",len(word2id))
    word2vecs = build_word2vec(word2vec_pretrained, word2id, save_to_path=None)
    print("length of word2vecs:", len(word2vecs))
    #训练数据
    traindata = CommentDataset(trainpath,word2id,id2word)
    trainloader = Data.DataLoader(traindata, batch_size, shuffle=True, num_workers=0, collate_fn=mycollate_fn)
    #验证数据
    validatedata = CommentDataset(validatepath, word2id, id2word)
    validateloader = Data.DataLoader(validatedata, batch_size, shuffle=True, num_workers=0, collate_fn=mycollate_fn)
    #测试数据
    testdata = CommentDataset(testpath, word2id, id2word)
    testloader = Data.DataLoader(testdata, batch_size, shuffle=False, num_workers=0, collate_fn=mycollate_fn)

    #step2: 建立模型
    # lstmmodel = LSTMModel(embedding_dim, hidden_dim, pre_weight=word2vecs)
    textcnnmodel = TextCNN(vocab_size, embedding_dim, filters_num=128, filter_size=[2,3,4,5], pre_weight=word2vecs)

    # #step3: 设置tensorboard可视化参数
    # visdir = time.strftime('assets/visualize/' + lstmmodel.modelName + '_%m%d_%H_%M')
    # trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    # validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    # testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))

    #step3: 训练模型
    text_trainedmodel = train(textcnnmodel, batch_size, lr, epochs, device, trainloader, validateloader)
    # lstm_trainedmodel = train(lstmmodel, batch_size, lr, epochs, device, trainloader, validateloader)

    # 加载valid上最优模型
    path = "/Users/cglin/Desktop/output/4/models/save.bin"
    test_model = TextCNN(vocab_size, embedding_dim, filters_num=128, filter_size=[2,3,4,5], pre_weight=word2vecs)
    test_model.load_state_dict(torch.load(args.path, map_location="cpu"), strict=False)
    #step4: 测试模型
    text_testacc = test(test_model, device, testloader)
    # lstm_testacc = test(lstm_trainedmodel, device, testloader)



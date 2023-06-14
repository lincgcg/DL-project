import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

# 初始化wanda
wandb.init(
    # set the wandb project where this run will be logged
    project="DL-project",
    name = "CNN-MNIST",
    # track hyperparameters and run metadata
    config={
    "Learning Rate": 0.2,
    "Batch size": 64,
    "model": "CNN"
    }
)


# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the network
model = Net().to(device)

# Specify the loss function
criterion = nn.CrossEntropyLoss()

# Specify the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Load the MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Training the Model
report_steps = 100
total_loss = 0
eopch_num = 10
for epoch in range(eopch_num):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        total_loss += loss.item()
        if (batch_idx + 1) % report_steps == 0:
            # logging.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
            print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, batch_idx + 1, total_loss / report_steps))
            wandb.log({"loss": total_loss / report_steps})
            total_loss = 0.0
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Testing the Model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    predicted_labels = []
    true_labels = []
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
            predicted_labels.extend(predictions.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        
        wandb.log({"Accuracy": accuracy})
        wandb.log({"Precision": precision})
        wandb.log({"Recall": recall})
        wandb.log({"'F1 Score":  f1})
        # print('Precision: ', precision)
        # print('Recall: ', recall)
        # print('F1 Score: ', f1)
        # print("==========================================")
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
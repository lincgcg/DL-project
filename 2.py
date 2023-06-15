import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import wandb
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

parser.add_argument("--name", type=str, default="CNN",
                        help="model name")

args = parser.parse_args()

# # 初始化wanda
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="DL-project-2",
#     name = args.name,
#     # track hyperparameters and run metadata
#     config={
#     "Learning Rate": args.lr,
#     "Batch size": 64,
#     "model": "CNN"
#     }
# )

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for the training and testing data
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training and testing datasets
train_data = datasets.ImageFolder(root='/Users/cglin/Desktop/data/2/train', transform=transform)
test_data = datasets.ImageFolder(root='/Users/cglin/Desktop/data/2/test', transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# Define a simple CNN
## shallow CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # Assuming there are 2 classes - cats and dogs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a deeper CNN
class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 2)  # Assuming there are 2 classes - cats and dogs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Evaluate the model
def check_accuracy(test_loader, model):
    correct = 0
    total = 0

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    print("Accuracy_Test")
    print(accuracy)
    print("Precision_Test")
    print(precision)
    print("Recall_Test")
    print(recall)
    print("F1_Test")
    print(f1)
    # wandb.log({"Accuracy_Test": accuracy})
    # wandb.log({"Precision_Test": precision})
    # wandb.log({"Recall_Test": recall})
    # wandb.log({"F1_Test":  f1})

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# Instantiate the CNN
model = DeeperNet()
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

report_steps = 30
total_loss = 0
epoch_num = 10
# Train the model
for epoch in range(epoch_num):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        if (i + 1) % report_steps == 0:
            # logging.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
            print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
            # wandb.log({"loss": total_loss / report_steps})
            total_loss = 0.0
    print("epoch num : ")
    print(epoch)
    check_accuracy(test_loader, model)

print("Final Test")
check_accuracy(test_loader, model)



print('Finished Training')



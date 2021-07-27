import torch
import torchvision
import argparse
from preprocessing.utils import globimgs
from models import *


# Command Line Argument Parser
parser = argparse.ArgumentParser(description="Train signauth model with tweaking hyper-parameters")
parser.add_argument("--classes", dest="classes", type=int, default=num_classes,
                    help=f"num of model out features (default: {num_classes})")
parser.add_argument("--epochs", dest="epochs", type=int, default=num_epochs,
                    help=f"number of epochs (default: {num_epochs})")
parser.add_argument("--batchsize", dest="batchsize", type=int, default=batch_size,
                    help=f"batchsize of the dataset (default: {batch_size})")
parser.add_argument("--learningrate", dest="learningrate", type=float, default=learning_rate,
                    help=f"learning rate of optimizer adam (default: {learning_rate})")
parser.add_argument("--update", dest="update", action="store_true", default=False,
                help=f"update num_classes in predict.py (default: False)")
args = parser.parse_args()

# Updating num_classes in predict.py
if args.update:
    with open("predict.py") as f:
        lines = []
        for line in f.readlines():
            if "num_classes =" in line:
                line = f"num_classes = {args.classes}\n"
            lines.append(line)

    with open("predict.py", "w") as f:
        f.writelines(lines)

# Hyper-parameters (Updated Through Command Line)
num_classes = args.classes
num_epochs = args.epochs
batch_size = args.batchsize
learning_rate = args.learningrate

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(device).upper()}\n")

# Preprocessing - Finding Total Images
train_count = len(globimgs("data/train", ["/**/*.jpg", "/**/*.png", "/**/*.jpeg"]))
test_count = len(globimgs("data/test", ["/**/*.jpg", "/**/*.png", "/**/*.jpeg"]))

# Dataset Load
train_dataset = torchvision.datasets.ImageFolder(root="data/train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="data/test", transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Building Model
model = ConvNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# Training And Testing Accuracy
def reducefloat(floatno, digits=4):
    return float(("{:." + str(digits) + "f}").format(floatno))

best_accuracy = 0.0
total_step = len(train_loader)

for epoch in range(num_epochs):
    model.eval()
    train_accuracy, test_accuracy, train_loss = 0.0, 0.0, 0.0
    
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))

        print(f"Epoch: [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {reducefloat(loss.item())}")

    train_accuracy = reducefloat(train_accuracy / train_count)
    train_loss = train_loss / train_count

    model.eval()
    
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
            
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
    
    test_accuracy = reducefloat(test_accuracy / test_count)

    print(f"Train Loss: {reducefloat(train_loss.item())}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")
    
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(),"models/model.pth")
        best_accuracy = test_accuracy

print(f"The maximum accuracy obtained by model is {best_accuracy}")

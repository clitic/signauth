import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image


# Hyper-parameters (Default)
num_classes = 2
num_epochs = 10
batch_size = 20
learning_rate = 0.001

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# CNN Neural Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32*128*128, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
            
        output = self.pool(output)
            
        output = self.conv2(output)
        output = self.relu2(output)
            
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32*128*128)    
        output = self.fc(output)
            
        return output
        
    def predict(self, model, device, img_path, classes, transform):
        img = Image.open(img_path)
        img = transform(img).float().unsqueeze(0)
        img = img.to(device)
        output = model(img)
        index = output.data.numpy().argmax()
        return classes[index]


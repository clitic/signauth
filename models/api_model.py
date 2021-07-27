import torch
from __init__ import *


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters (Update)
num_classes = num_classes

# Loading CNN Model
checkpoint = torch.load("model.pth")
model = ConvNet(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint)
model.eval()

# Saving jitted model 
traced_cell = torch.jit.trace(model, torch.randn(1, 3, 256, 256))
torch.jit.save(traced_cell, "../api/model.pth")

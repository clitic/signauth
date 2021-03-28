import torch
import pathlib
import csv
from preprocessing.utils import globimgs
from models import *


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyper-parameters (Update)
num_classes = num_classes


# Loading CNN Model
checkpoint = torch.load("models/model.pth")
model = ConvNet(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint)
model.eval()


# Getting images paths and classes
classes = sorted([folder.name.split("/")[-1] for folder in pathlib.Path("data/train").iterdir()])
images_path = globimgs("data/predict", ["/*.jpg", "/*.png", "/*.jpeg"])
images_name = map(lambda x: x.replace("data/predict/", ""), globimgs("data/predict", ["/*.jpg", "/*.png", "/*.jpeg"]))
total_images = len(images_path)


# Generating predictions.csv at temp folder
print("{:<10} {:<10} {:<40} {:<0}".format("Progress", "S.No.", "Image Name", "Prediction"))
with open("temp/predictions.csv", "w", newline="") as f:
    wrt = csv.writer(f)
    wrt.writerow("image_path prediction".split())

    for i, (img_path, img_name) in enumerate(zip(images_path, images_name)):
        prediction = model.predict(model, device, img_path, classes=classes, transform=transform)
        print("{:<10} {:<10} {:<40} {:<0}".format(f"{int(((i+1)/total_images)*100)}%", i+1, img_name, prediction))
        wrt.writerow([img_path, prediction])


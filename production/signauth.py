from torch import jit as torchjit
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import os


os.chdir(os.path.split(__file__)[0])


def processimage(img_path, scan=False):

	def scaneffects(img):
	    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
	    bg_img = cv2.medianBlur(dilated_img, 15)
	    diff_img = 255 - cv2.absdiff(img, bg_img)
	    norm_img = diff_img.copy() 
	    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
	    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	    return thr_img


	img = Image.open(img_path)
	img.convert("LA")
	width, height = img.size
	cutfactor = max(img.size)
	left, top, right, bottom = int((width-cutfactor)/2), int((height-cutfactor)/2), int((width+cutfactor)/2), int((height+cutfactor)/2)
	bg = Image.new("RGB", (right-left, bottom-top), (255, 255, 255))
	bg.paste(img, (-left, -top))
	img = bg
	img.thumbnail((256, 256), Image.ANTIALIAS)
	img.save("./temp.jpg")

	if scan:
		img = cv2.imread("./temp.jpg")
		cv2.imwrite("./temp.jpg", scaneffects(img))

	return Image.open("./temp.jpg")


# CLI Parser
parser = argparse.ArgumentParser(description="Production build for signauth, predict signature styles that are real or fake.")
parser.add_argument("image", type=str,
                    help="path of signature image")
parser.add_argument("--scan", dest="scan", action="store_true", default=False,
                    help="add a scan filter to image while processing (default: False)")
args = parser.parse_args()


# Image Transformations	
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# Loading Jitted Model
model = torchjit.load("modeljit.pth")
model.eval()


# Making Predictions
classes = ["fake", "real"]
img = processimage(args.image, scan=args.scan)
img = transform(img).float().unsqueeze(0)
output = model(img)
index = output.data.numpy().argmax()

print(classes[index])

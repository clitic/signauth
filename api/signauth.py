from torch import jit
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import os


# Image Transformations	
transform = transforms.Compose([
	transforms.Resize((256, 256)),
	# transforms.RandomVerticalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	])


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

	new_image_path, _ = os.path.splitext(img_path)
	new_image_path += "_signauth_processed.jpg"
	img.save(new_image_path)

	if scan:
		img = cv2.imread(new_image_path)
		cv2.imwrite(new_image_path, scaneffects(img))

	return Image.open(new_image_path)


def main():
	try:
		os.chdir(os.path.split(__file__)[0])
	except:
		pass

	# CLI Parser
	parser = argparse.ArgumentParser(description="production build for signauth, predict signature styles that are real or fake.")
	parser.add_argument("image", type=str,
						help="path of signature image")
	parser.add_argument("--model_pth", dest="model_pth", default="model.pth",
						help="signauth jitted model path (default: model.pth)")
	parser.add_argument("--scan", dest="scan", action="store_true", default=False,
						help="add a scan filter to image while processing (default: False)")
	args = parser.parse_args()

	# Loading Jitted Model
	model = jit.load(args.model_pth)
	model.eval()

	# Making Predictions
	classes = ["fake", "real"]
	img = processimage(args.image, scan=args.scan)
	img = transform(img).float().unsqueeze(0)
	output = model(img)
	index = output.data.numpy().argmax()

	print(classes[index])

if __name__ == "__main__":
	main()

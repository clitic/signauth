import glob
import cv2
import numpy as np


def globimgs(path, globs:list):
	"""returns a list of files with path with globing with more than one extensions"""

	imgs = []
	for i in globs:
		imgs.extend(glob.glob(path + i))

	paths = []
	for path in imgs:
		paths.append(path.replace("\\", "/"))
	return paths


def scaneffects(img):
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 15)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() 
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return thr_img


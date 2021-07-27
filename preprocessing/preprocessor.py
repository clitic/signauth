import argparse
from PIL import Image
import shutil
import os
import colorama
from utils import *


colorama.init(autoreset=True)

try:
    os.chdir(os.path.split(__file__)[0])
except:
    pass

def processimages(iterable:list, scan=False):
    """
    resize images to 256x256 pixels and grayscale them

    Args:
        iterable (list): list with path of images
        method (str): resize images

    Methods:
        reshape: resize images with respect to aspect ratio
        compress: resize images by compressing it from sides
        centercrop: crop image center part of pixels 256x256 (ascept ratio mantained)
    """

    total_images = len(iterable)

    for i, img_path in enumerate(iterable):
        img = Image.open(img_path)
        img.convert("LA")
        filename = os.path.splitext(img_path)[0]
        new_filename = filename + ".jpg"

        width, height = img.size
        cutfactor = max(img.size)
        left, top, right, bottom = int((width-cutfactor)/2), int((height-cutfactor)/2), int((width+cutfactor)/2), int((height+cutfactor)/2)
        bg = Image.new("RGB", (right-left, bottom-top), (255, 255, 255))
        bg.paste(img, (-left, -top))
        img = bg
        img.thumbnail((256, 256), Image.ANTIALIAS)

        try:
            os.remove(img_path)
            print(colorama.Fore.GREEN + f"[{int(((i+1)/total_images)*100)}%]", f"image removed {img_path}")
        except:
            print(colorama.Fore.RED + "[!]", f"image removal {img_path} failed")
            if True if "y" == (input(f"File {img_path} couldn't be removed\nDo you want to continue [y/n]? ")).lower() else False:
                pass
            else:
                break

        img.save(new_filename)

        if scan:
            img = cv2.imread(new_filename)
            
            try:
                os.remove(new_filename)
                print(colorama.Fore.GREEN + f"[{int(((i+1)/total_images)*100)}%]", f"image removed {new_filename}")
            except:
                print(colorama.Fore.RED + "[!]", f"image removal {new_filename} failed")
                if True if "y" == (input(f"File {new_filename} couldn't be removed\nDo you want to continue [y/n]? ")).lower() else False:
                    pass
                else:
                    break

            cv2.imwrite(new_filename, scaneffects(img))

        print(colorama.Fore.GREEN + f"[{int(((i+1)/total_images)*100)}%]", f"image saved {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images by grayscaling and resizing to 256x256 for signauth")
    parser.add_argument("--process", dest="process", type=str, default="train,test",
                        help="images to be processed (default: train,test)")
    parser.add_argument("--scan", dest="scan", action="store_true", default=False,
                        help="add a scan filter to images (default: False)")
    parser.add_argument("--backup", dest="backup", action="store_true", default=False,
                        help="backup data folder in temp folder before preprocessing (default: False)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False,
                        help="overwrite previous backuped data.zip (default: False)")
    args = parser.parse_args()

    # Backup Data
    if args.backup:
        if os.path.exists("../temp/data.zip"):
            if args.overwrite:
                print(colorama.Fore.CYAN + "[INFO]", "Backing up data")
                shutil.make_archive("../temp/data", "zip", "../data")
                print(colorama.Fore.GREEN + "[√]", "Backup successful data.zip overwritted")
            else:
                print(colorama.Fore.CYAN + "[INFO]", "Overwrite data.zip skipped")
        else:
            print(colorama.Fore.CYAN + "[INFO]", "Backing up data")
            shutil.make_archive("../temp/data", "zip", "../data")
            print(colorama.Fore.GREEN + "[√]", "Backup successful data.zip created")
    else:
        print(colorama.Fore.RED + "[!]", "Backing up data skipped")

    # Processing Images
    if "train" in args.process:
        train_imgs = globimgs("../data/train", ["/**/*.jpg", "/**/*.png", "/**/*.jpeg"])
        processimages(train_imgs, scan=args.scan)

    if "test" in args.process:
        test_imgs = globimgs("../data/test", ["/**/*.jpg", "/**/*.png", "/**/*.jpeg"])
        processimages(test_imgs, scan=args.scan)

    if "predict" in args.process:
        predict_imgs = globimgs("../data/predict", ["/*.jpg", "/*.png", "/*.jpeg"])
        processimages(predict_imgs, scan=args.scan)

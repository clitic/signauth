# SignAuth - Handwritten Signatures Authentication

<p align="center">
  <img src="https://raw.githubusercontent.com/360modder/signauth/master/others/sample.jpg">
  <a href="https://colab.research.google.com/github/360modder/signauth/blob/master/others/SignAuth_Google_Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

*signauth* is a deep learning model built using pytorch which can authenticate handwritten signatures using cnn image classification. Model can differentiate between fake and real handwritten signatures. Model needs labels for every signatures (*supervised learning*). For more precision model uses image of **256x256** pixels, this cause in higher model sizes but accuracy is more.

Model was able to get 98.8999% accuracy on test images. You can download that model from [google drive](https://drive.google.com/uc?export=download&id=1MB6FgraqQpwXP9E8kHlgW3qHJpxUcj6P).

The development handwritten signatures dataset is used from this [kaggle](https://www.kaggle.com/divyanshrai/handwritten-signatures "dataset") dataset.

## Getting Started

Default images are labeled as XXXYYZZZ

- XXX -> person id who signed
- YY -> number of sign
- ZZZ -> real person id who has this signature style

So if XXX equal ZZZ then it is a real signature, and if XXX notequal to ZZZ then it is a fake signature.

- Intitial Setup

```bash
git clone https://www.github.com/360modder/signauth.git
cd signauth
pip install -r requirements.txt
```

- Pre-Trained Model

```bash
python api/signauth.py <image path> --scan
```

- Training & Predicting

```bash
python preprocessing/preprocessor.py --process train,test,predict --backup --overwrite
python train.py
python predict.py
```

The main model.py or the model class is [models/__init__.py](models/__init__.py).

## Training

Training model is very easy, you have to place your images in following steps.

1. Gather signature images (prefer size 256x256)
2. Create a folder and name it as same as the label for sign image
3. Place all image files in that folder
4. Place this folder inside **data/train/[foldername i.e. label]/**
5. Do same or split files from there and do same for test images

If images are non-uniform you can run.

```bash
python preprocessing/preprocessor.py --backup
```

> Note: Use **--scan** flag in above command to apply scan effect to signature images for more precision in some cases.

After prepration of data you run in command line.

```bash
python train.py --classes 2 --update
```


## Predictions (Classifications)

Once you train a model, then you have to place your source images (256x256) in **data/predict** directory the run command.

```bash
python predict.py
```

If images are not of 256x256 or any other issue you can place them inside **data/predict** folder then run this command.

```bash
python preprocessing/preprocessor.py --process predict --scan --backup
```

This will reshape images into ascpect ratio 1:1 and resize them to 256x256 with a scan filter for model.


## API Model Build

For building model for api can be done by jitting the main model using a standard input and then used. Building of one such model can be done by the following command.

```bash
python models/api_model.py
```

> Note : You must run this script after training a model and getting an model.pth file in models folder.

To run fastapi server on localhost first change directory to api and then run this command. 

```bash
uvicorn main:app --reload
```

Test API by uploading local image.

```python
import requests

files = {
  "file": ("00503005.png", open("00503005.png", "rb"), "image/png")
}

response = requests.post("http://127.0.0.1:8000/upload_image", files=files)
print(response.json())
```

## Tensorflow Lite Model

Once you have trained a model it is generated in models directory under name *model.pth*. To convert this model to tensorflow lite model first install all *commented* dependencies in [requirements.txt](requirements.txt) then run the following commands.

The model conversion chain is followed as *pytorch -> onnx -> tf -> tflite*.

```bash
python models/tflite.py
```

This command will create a tflite model in models directory. You can further tweak models/tflite.py for more advanced conversions.


## TODO

- fixing predictions for an un-classified sign
- more correct and precise predictions (classifications)


## License

© 2021 360modder

This repository is licensed under the MIT license. See LICENSE for details.

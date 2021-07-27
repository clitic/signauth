import torch
import tensorflow as tf
import onnx
import onnx_tf
import os, shutil
from PIL import Image
from __init__ import *

 
try:
	os.chdir(os.path.split(__file__)[0])
except:
	pass

if os.path.exists("temp"):
	shutil.rmtree("temp")

os.mkdir("temp")

pytorch_model = ConvNet(num_classes=num_classes)
pytorch_model.load_state_dict(torch.load("model.pth"))
 
# dummy_input = Image.open("../temp/dummy.jpg")
# dummy_input = transform(dummy_input).float().unsqueeze(0)
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(pytorch_model, dummy_input, "temp/model.onnx")
 
onnx_model = onnx.load("temp/model.onnx")
onnx.checker.check_model(onnx_model)
 
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph("temp/model.pb") 

convertor = tf.lite.TFLiteConverter.from_saved_model("temp/model.pb")
tflite_model = convertor.convert()

with open("model.tflite", "wb") as f:
	f.write(tflite_model)
	
# sunprocess.run("tflite_convert --saved_model_dir ../temp/model.pb --output_file model.tflite")
os.remove("temp/model.onnx")
shutil.rmtree("temp/model.pb")

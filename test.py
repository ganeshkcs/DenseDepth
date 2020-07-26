import os
import glob
import argparse
import matplotlib
import numpy as np
import timeit
import zipfile
import datetime

from multiprocessing import Process

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_zip_images, save_output, scale_up
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--bgNumber', default='1', type=int, help='current bg number')
parser.add_argument('--output', default='output/', type=str, help='Output folder.')
parser.add_argument('--start', default=0, type=int, help='starting index.')
parser.add_argument('--end', default=10, type=int, help='ending index.')
# parser.add_argument('--fileNameCounter', default=1, type=int, help='ending index.')
args = parser.parse_args()
model = 'nyu.h5'
# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

model = args.model
input_dir = args.input
output_dir = args.output
start = args.start
end = args.end
bg_number = args.bgNumber
file_name_counter = (bg_number - 1) * 4000 + 1

#Load model into GPU / CPU
model = load_model(model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(model))
mylist = []
# mylist = [f for f in glob.glob(input_dir)]
for count in range(1,101):
  zip_file_name = "/content/drive/My Drive/Utils/S15A/Images/fg_bg" + str(count) + ".zip"
  my_dict = {"bg_number": count, "zip_file_name" : zip_file_name}
  mylist.append(my_dict)

print("My_List", mylist[start:end])

for file_name in mylist[start:end]:
  print(file_name, start, end)
  print("Bg Loop",datetime.datetime.now())
  with zipfile.ZipFile(file_name["zip_file_name"], 'r') as zip:
    start = timeit.default_timer()
    print("start", start)
    file_list = zip.namelist()
    new_zip_name = file_name["bg_number"]
    dense_depth_zip = zipfile.ZipFile(output_dir+f'/fb_bg_depth{new_zip_name}.zip', mode='a', compression=zipfile.ZIP_STORED)
    for i in range(0, 4000, 50):
      snipped_list = file_list[i:i+50]
      inputs= load_zip_images(zip, snipped_list)
      inputs = scale_up(2, inputs)
      outputs = predict(model, inputs)
      save_output(outputs, output_dir, snipped_list, dense_depth_zip, is_rescale=True)
      
    stop = timeit.default_timer()
    execution_time = stop - start
    dense_depth_zip.close()
    print("Program Executed in "+str(execution_time))



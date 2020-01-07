from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # works on windows 10 to force it to use CPU

from data import *

RunWithGPU = True
PerformTraining = False

if RunWithGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# tflite_convert --output_file=unet_Solar.tflite --keras_model_file=unet_Solar.hdf5
test_single_image = test_image_prep('1.png')
test_lite_img = load_model_lite_single_predict('unet_Solar.tflite', test_single_image)

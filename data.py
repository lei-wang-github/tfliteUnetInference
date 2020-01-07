from __future__ import print_function
import numpy as np 

import skimage.io as io
import skimage.transform as trans
import time
import tflite_runtime.interpreter as tflite
from PIL import Image

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

        
def test_image_prep(image_file_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    img = io.imread(image_file_path, as_gray=as_gray)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def load_model_lite_single_predict(model_path, tf_image):
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(tf_image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    t1 = time.time()
    interpreter.invoke()
    print("elapsed-time =", time.time() - t1)
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    outputs = interpreter.get_tensor(output_details[0]['index'])
    
    output = outputs[0]
    img = output[:,:,0]
    io.imsave('solar.png', img)
    return img

        

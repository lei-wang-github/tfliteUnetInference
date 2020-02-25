import numpy as np

import skimage.io as io
import skimage.transform as trans
import time
import tflite_runtime.interpreter as tflite

img_height = 512
img_width = 512

def test_image_prep(image_file_path, target_size=(img_height, img_width), flag_multi_class=False, as_gray=True):
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
    img_float32 = output[:,:,0]
    img_float32 = img_float32 * 255
    img_uint8 = img_float32.astype(np.uint8)

    io.imsave('out.png', img_uint8)
    return img_uint8

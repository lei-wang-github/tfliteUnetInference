from data import *

# To convert to tflite model from Keras: tflite_convert --output_file=unet_xxx.tflite --keras_model_file=unet_xxx.hdf5
test_single_image = test_image_prep('0.png')
test_lite_img = load_model_lite_single_predict('unet_gscpLaser228x448-rd32.tflite', test_single_image)

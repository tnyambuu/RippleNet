import tensorflow as tf

# List physical devices available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is NOT using the GPU.")

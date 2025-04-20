import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Create a simple TensorFlow computation to test GPU usage
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[2.0, 0.0], [1.0, 3.0]])

with tf.device('/GPU:0'):
    result = tf.matmul(a, b)

print("Computation Result:\n", result.numpy())
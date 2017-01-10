from __future__ import division, print_function, absolute_import
import tensorflow as tf 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


#read the image 
img = Image.open('PTZfish.tiff')
img.seek(0)
n=img.n_frames

a,b=(np.array(img)).shape
imarray=np.zeros((n,a,b))

for i in range(n):
    try:
        img.seek(i)
        #print img.getpixel( (1, 0))
        imarray[i]=np.array(img)
        #print(imarray)
    except EOFError:
        # Not enough frames in img
        break

#transfer into a 2D array where each row is one frame
im_input= imarray.reshape((n,-1)) 
print(im_input.shape)
print(im_input[1])


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 1000 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 262144# data input (img shape: x*y, here 262144= 512*512)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

print("now launch the graph")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    total_batch = int(n/batch_size)

    
    # Training cycle
    print("training cycle")
    for epoch in range(training_epochs):
        # Loop over all batches
        print(epoch)
        for i in range(total_batch):
            print(i)
            batch_xs= im_input[np.random.randint(0,im_input.shape[0],batch_size)]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: im_input[0:examples_to_show,:]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(im_input[i], (a, b)))
        a[1][i].imshow(np.reshape(encode_decode[i], (a, b)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
    


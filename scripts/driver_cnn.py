
# coding: utf-8


import tensorflow as tf
import time

## IMPORT FROM DATASET
import load_image_batch as driving_data


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Input and output variables

INPUTS = 60 * 200 * 3
OUTPUTS = 1
BATCH_SIZE = 20
NUM_EPOCHS = 50
LEARNING_RATE = 1e-04
NUM_IMAGES = 1327


try:
    sess.close()
except:
    pass

sess = tf.InteractiveSession()


# Input and output placeholder
x_image = tf.placeholder(tf.float32, shape=[None, 64, 224, 3])
y = tf.placeholder(tf.float32, shape=[None, OUTPUTS])
pkeep = tf.placeholder(tf.float32)


# First Convolutional Layer
W_conv_1 = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.0, shape=[64]))
h_conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)
h_pool_1 = max_pool_2x2(h_conv_1)

print(h_pool_1.get_shape())

# Second Convolutional Layer
W_conv_2 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.0, shape=[128]))
h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
h_pool_2 = max_pool_2x2(h_conv_2)

print(h_pool_2.get_shape())

# Third Convolutional Layer
W_conv_3 = tf.Variable(tf.truncated_normal([2, 2, 128, 256], stddev=0.1))
b_conv_3 = tf.Variable(tf.constant(0.0, shape=[256]))
h_conv_3 = tf.nn.relu(conv2d(h_pool_2, W_conv_3) + b_conv_3)
h_pool_3 = max_pool_2x2(h_conv_3)

print(h_pool_3.get_shape())

# Fourth Convolutional Layer
W_conv_4 = tf.Variable(tf.truncated_normal([2, 2, 256, 512], stddev=0.1))
b_conv_4 = tf.Variable(tf.constant(0.0, shape=[512]))
h_conv_4 = tf.nn.relu(conv2d(h_pool_3, W_conv_4) + b_conv_4)
h_pool_4 = max_pool_2x2(h_conv_4)

print(h_pool_4.get_shape())

# Densely connected layer
W_fc1 = tf.Variable(tf.truncated_normal([4 * 14 * 512, 128], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.0, shape=[128]))
h_poolfc1_flat = tf.reshape(h_pool_4, [-1, 4 * 14 * 512])
h_fc1 = tf.nn.relu(tf.matmul(h_poolfc1_flat, W_fc1) + b_fc1)

# Densely connected layer
W_fc2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.0, shape=[256]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Dropout
h_drop = tf.nn.dropout(h_fc2, pkeep)

# Read out Layer
W_fc3 = tf.Variable(tf.truncated_normal([256, OUTPUTS], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.0, shape=[OUTPUTS]))
y_logits = tf.matmul(h_drop, W_fc3) + b_fc3

loss = tf.sqrt(tf.reduce_mean(tf.square(y - y_logits)), name='RMSE')
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train_step = optimizer.minimize(loss)

mse = tf.reduce_mean(tf.square(y_logits - y))

init = tf.global_variables_initializer()

# op to write model to Tensorboard
save_path = './model/'
saver = tf.train.Saver()

sess.run(init)

loss_train_array = []
test_accuracy_array = []
train_accuracy_array = []

for current_epoch in range(NUM_EPOCHS):
    
    test_dataset, test_labels = driving_data.get_test_batch(BATCH_SIZE)

    start = time.time()

    for step in range(int(NUM_IMAGES / BATCH_SIZE)):

        train_dataset, train_labels = driving_data.get_train_batch(BATCH_SIZE)
        
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph is should be fed to.
        feed_dict = {x_image: train_dataset, y: train_labels, pkeep: 0.8}
        _, loss_train = sess.run([train_step, loss],
                                 feed_dict=feed_dict)

        print("--- %s seconds ---" % (time.time() - start))

    print("Predictions: ")
    print("-------LABEL------:")
    print(test_labels)
    predictions = sess.run(
        y_logits, feed_dict={x_image: test_dataset, pkeep: 1.0})
    print("-------PREDICTIONS-------:")
    print(predictions)

    # We calculate the accuracies to plot their values later
    loss_train_array.append(loss_train)

    train_accuracy = sess.run(
        mse, feed_dict={x_image: train_dataset, y: train_labels, pkeep: 1.0})

    train_accuracy_array.append(train_accuracy)

    test_accuracy = sess.run(
        mse, feed_dict={x_image: test_dataset, y: test_labels, pkeep: 1.0})

    test_accuracy_array.append(test_accuracy)

    print (
        'Epoch %04d, '
        'loss train %.8f, train accuracy %.8f, test accuracy %.8f'
        %
        (current_epoch,
         loss_train,
         train_accuracy,
         test_accuracy))

filename = saver.save(sess, 'model/model.ckpt')
print("Model saved in file: %s" % filename)





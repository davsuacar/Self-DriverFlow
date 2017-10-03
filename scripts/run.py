import tensorflow as tf
import numpy as np
import cv2
import scipy.ndimage
import rospy
from geometry_msgs.msg import Twist

from image_preprocess import process_image

def shutdown(self):
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    self.cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)

def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Network variables
OUTPUTS = 3

try:
    sess.close()
except:
    pass

sess = tf.InteractiveSession()


# Input and output placeholder
x_image = tf.placeholder(tf.float32, shape=[None, 64, 224, 1])
y = tf.placeholder(tf.float32, shape=[None, OUTPUTS])
pkeep = tf.placeholder(tf.float32)


# First Convolutional Layer
W_conv_1 = tf.Variable(tf.truncated_normal([2, 2, 1, 64], stddev=0.1))
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
y_softmax = tf.nn.softmax(y_logits)

correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y, 1))

init = tf.global_variables_initializer()

# op to write model to Tensorboard
saver = tf.train.Saver()
saver.restore(sess, "./model/model.ckpt")

video_capture = cv2.VideoCapture(0)

# initiliaze
rospy.init_node('GoForward', anonymous=False)

# tell user how to stop TurtleBot
rospy.loginfo("To stop TurtleBot CTRL + C")

# What function to call when you ctrl + c
#rospy.on_shutdown(shutdown)

# Create a publisher which can "talk" to TurtleBot and tell it to move
# Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

# TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
r = rospy.Rate(10)

for i in range(10):
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    image = process_image(frame)
    image = np.array(scipy.misc.imresize(image, [64, 224]) / 255.0).reshape(64, 224, 1)

    predict = y_softmax.eval(feed_dict={x_image: [image], pkeep: 1})
    print(predict)

    # Twist is a datatype for velocity
    move_cmd = Twist()
    # let's go forward at 0.2 m/s
    move_cmd.linear.x = 0.1
    # let's turn at 0 radians/s
    move_cmd.angular.z = 5

    cmd_vel.publish(move_cmd)

video_capture.release()

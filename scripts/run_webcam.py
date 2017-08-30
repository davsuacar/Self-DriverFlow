import cv2
import rospy
from geometry_msgs.msg import Twist
from time import sleep
    
def callback(data):
	rospy.loginfo("I heard %s",data)

def listener():
	rospy.init_node('node_name2')
	rospy.Subscriber("/mobile_base/commands/velocity", Twist, callback)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

listener()
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read() 
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")

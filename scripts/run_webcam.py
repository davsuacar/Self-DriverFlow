import cv2
import rospy
from geometry_msgs.msg import Twist
from time import sleep
import csv
    
i = 0
vc = cv2.VideoCapture(0)

def callback(data):
	print("callback called...")
	global i
	global vc
	rospy.loginfo("I heard %s",data)
	with open('dataset.csv', 'a') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=' ',
		                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow([data.linear.z])

	    if vc.isOpened(): # try to get the first frame
	        rval, frame = vc.read()
            else:
		rval = False


	    rval, frame = vc.read()
	    cv2.imwrite("images/image_" + str(i) + ".png", frame)
	    i += 1 

def listener():
	rospy.init_node('node_name3')
	rospy.Subscriber("/mobile_base/commands/velocity", Twist, callback)



listener()
print("listener initialized...")

while True:
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")

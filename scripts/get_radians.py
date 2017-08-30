import rospy
from geometry_msgs.msg import Twist
from time import sleep
    
def callback(data):
	rospy.loginfo("I heard %s",data)

def listener():
	rospy.init_node('node_name2')
	rospy.Subscriber("/mobile_base/commands/velocity", Twist, callback)
	sleep(1)
	# spin() simply keeps python from exiting until this node is stopped
	# rospy.spin() 

listener()  

#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

def callback(data):
    rospy.loginfo("Received an image with height: %d, width: %d", data.height, data.width)

def listener():
    rospy.Subscriber("/rrbot/camera1/image_raw", Image, callback)

def move():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    rate = rospy.Rate(2)
    move = Twist()
    move.linear.x = 0.5
    move.linear.y = 0.0
    move.linear.z = 0.0
    move.angular.x = 0.0
    move.angular.y = 0.0
    move.angular.z = 0.5

    rospy.loginfo("Publishing velocity commands to /cmd_vel")

    while not rospy.is_shutdown():
        pub.publish(move)
        rate.sleep()

def main():
    rospy.init_node('robot_controller', anonymous=True)
    listener()
    try:
        move()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('enph353_ros_lab')
import sys
import rospy
import cv2
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

## @class image_converter
#  @brief A ROS node that subscribes to an image stream, processes the image to detect a line, and publishes velocity commands to follow the line.
class image_converter:
    ## @brief Constructor: Initializes the image converter, subscribes to the camera topic, and sets up the velocity publisher.
    def __init__(self):
        # Initialize the CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Subscribe to the camera image topic
        self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)
        
        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    ## @brief Callback function that processes the incoming image and extracts the line.
    #  @param data The incoming image data from the camera.
    def callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV grayscale image
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        
        # Create a binary mask to detect the dark grey line
        mask = cv2.inRange(cv_image, 0, 75)
        
        # Get image height
        image_height = mask.shape[0]
        
        # Define the row of interest near the bottom of the image
        index = image_height - 3
        xmin = 0
        xmax = mask.shape[1] - 1
        
        # Scan for the left and right edges of the detected line
        for col in range(1, mask.shape[1] - 1):
            if mask[index, col] == 255 and mask[index, col-1] == 0:
                xmin = col  # Left edge of the line
            
            if mask[index, col] == 255 and mask[index, col+1] == 0:
                xmax = col  # Right edge of the line
        
        # Compute the center of the detected line
        xavg = (xmax + xmin) // 2
        
        # Get the image width
        image_width = mask.shape[1]
        
        # Calculate the error from the center of the image
        err = xavg - image_width // 2
        
        # Publish velocity command based on the error
        self.publish_velocity(err)

    ## @brief Computes and publishes velocity commands based on the detected line position error.
    #  @param error The difference between the line position and the center of the image.
    def publish_velocity(self, error):
        # Create a Twist message for velocity commands
        vel_msg = Twist()
        vel_msg.linear.x = 0.2  # Constant forward speed
        vel_msg.angular.z = -float(error) / 100.0  # Proportional control for steering
        
        # Publish the velocity command
        self.vel_pub.publish(vel_msg)
        rospy.loginfo("Published velocity command: linear.x=%.2f, angular.z=%.2f", vel_msg.linear.x, vel_msg.angular.z)

## @brief Main function to initialize the ROS node and start the image processing.
#  @param args Command-line arguments.
def main(args):
    # Initialize the ROS node
    rospy.init_node('image_converter', anonymous=True)
    
    # Create an instance of the image converter class
    ic = image_converter()
    
    try:
        # Keep the node running
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    # Destroy any OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

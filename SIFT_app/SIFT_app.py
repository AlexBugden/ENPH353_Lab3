#!/usr/bin/env python3

# Import necessary libraries
from PyQt5 import QtCore, QtGui, QtWidgets  
from python_qt_binding import loadUi  
import cv2  
import sys  
import numpy as np  

# Main application class
class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        # Load the UI file
        loadUi("./SIFT_app.ui", self)

        # Camera settings
        self._cam_id = 0  # Default camera ID (usually the built-in webcam)
        self._cam_fps = 10  # Frames per second for the camera
        self._is_cam_enabled = False  # Flag to check if the camera is enabled
        self._is_template_loaded = False  # Flag to check if a template image is loaded

        # Connect button signals to their respective slots
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Initialize the camera device
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)  # Set camera width to 320 pixels
        self._camera_device.set(4, 240)  # Set camera height to 240 pixels

        # Timer to trigger camera frame capture at regular intervals
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)  # Set timer interval based on FPS

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

    # Slot for the browse button to load a template image
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()  # Open a file dialog
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)  # Allow selecting existing files
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]  # Get the selected file path

        # Load the template image in grayscale
        self.template_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        # Resize the template image if it's larger than 320x240
        x, y = self.template_image.shape
        if x > 320 or y > 240:
            self.template_image = cv2.resize(self.template_image, (320, 240))

        # Convert the template image to a QPixmap and display it in the UI
        pixmap = self.convert_cv_to_pixmap(self.template_image)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

        # Detect keypoints and descriptors in the template image using SIFT
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.template_image, None)
        self._is_template_loaded = True  # Set flag to indicate template is loaded

    # Helper function to convert an OpenCV image to a QPixmap for display in the UI
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    # Slot to query the camera and process the captured frame
    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()  # Capture a frame from the camera
        if not ret:
            print("Error: Could not capture frame from camera.")
            return

        # Convert the captured frame to grayscale for SIFT processing
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If no template is loaded, just display the grayscale frame
        if not self._is_template_loaded:
            pixmap = self.convert_cv_to_pixmap(cv2.cvtColor(grayframe, cv2.COLOR_GRAY2BGR))
            self.live_image_label.setPixmap(pixmap)
            return

        # Detect keypoints and descriptors in the grayscale frame using SIFT
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

        # Check if descriptors are found in both the template and the frame
        if desc_grayframe is None or self.desc_image is None:
            print("Warning: No descriptors found.")
            return

        # FLANN-based matcher for feature matching
        index_params = dict(algorithm=0, trees=5)  # FLANN parameters
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        # Filter good matches using the Lowe's ratio test
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # Keep matches with a good distance ratio
                good_points.append(m)

        # Convert the grayscale frame to color for drawing
        color_frame = cv2.cvtColor(grayframe, cv2.COLOR_GRAY2BGR)

        # If enough good matches are found, compute homography and draw a bounding box
        if len(good_points) > 12:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            if matrix is not None:
                h, w = self.template_image.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                color_frame = cv2.polylines(color_frame, [np.int32(dst)], True, (255, 0, 0), 3)

            # Display the frame with the bounding box
            pixmap = self.convert_cv_to_pixmap(color_frame)
            self.live_image_label.setPixmap(pixmap)
        else:
            # If not enough matches, display the template and frame side by side with matches drawn
            h, w = frame.shape[:2]
            h2, w2 = self.template_image.shape[:2]
            combined_width = w + w2
            combined_image = np.zeros((h, combined_width, 3))

            template_color = cv2.cvtColor(self.template_image, cv2.COLOR_GRAY2BGR)
            
            combined_image[0:h2, 0:w2] = template_color
            combined_image[0:h, (combined_width - w):combined_width] = color_frame
            
            matches_img = cv2.drawMatches(template_color, self.kp_image, color_frame, kp_grayframe, good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
            pixmap = self.convert_cv_to_pixmap(matches_img)
            self.live_image_label.setPixmap(pixmap)

    # Slot to toggle the camera on and off
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()  # Stop the timer (disable camera)
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()  # Start the timer (enable camera)
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

# Main entry point of the application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
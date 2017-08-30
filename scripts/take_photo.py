import numpy as np
import cv2
import scipy.misc
import scipy.ndimage

video_capture = cv2.VideoCapture(0)
print(video_capture)

for i in range(3):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    print(frame)

video_capture.release()

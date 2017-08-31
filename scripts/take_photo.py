import numpy as np
import cv2
import scipy.misc
import scipy.ndimage

video_capture = cv2.VideoCapture(0)
print(video_capture)

i = 0
for i in range(3):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imwrite("image_" + str(i) + ".png", frame)
    i += 1
video_capture.release()

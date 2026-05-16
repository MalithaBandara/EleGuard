"""
Helper script to output the FPS and total frame count of a test video file.
Uses OpenCV to extract video properties.
"""
import cv2
cap = cv2.VideoCapture("c:/EleGuard/test/test2.mp4")
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Total Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

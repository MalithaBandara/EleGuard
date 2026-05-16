"""
Helper script to verify the presence of an audio track in a test video file.
Uses moviepy to attempt to extract and check the audio track.
"""
from moviepy import VideoFileClip
import sys

try:
    clip = VideoFileClip("c:/EleGuard/test/test2.mp4")
    if clip.audio is not None:
        print("Audio is present")
    else:
        print("No audio track found")
    clip.close()
except Exception as e:
    print("Error:", e)

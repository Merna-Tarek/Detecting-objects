#Real-Time Object Detection using ORB and OpenCV

This project implements a simple, efficient real-time object detection system using the ORB (Oriented FAST and Rotated BRIEF) feature detector and OpenCV. It matches features in a live video feed from a webcam against a template image to detect the presence of a specific object.

Features:
- Real-time object detection using webcam feed
- ORB (Oriented FAST and Rotated BRIEF) feature detector for efficient matching.
- Adjustable threshold for detecting specific objects based on feature matches.
- CPU-friendly, works without additional libraries like CUDA or GPU processing.

Installation
- OpenCV: For image processing and object detection.
- NumPy: For numerical operations.

Explanation:
1- ORB Detector: The ORB detector detects and describes key points in the live video feed and the template image.
2- Brute-Force Matcher: The detected key points are matched using the Brute-Force Matcher with Hamming distance for binary descriptors.
3- Threshold-Based Detection: When the number of feature matches exceeds a defined threshold, the program recognizes the object as detected.
4- Feedback in Video Feed: A rectangle is drawn around the detected object in the video feed, with the match count and detection message displayed.


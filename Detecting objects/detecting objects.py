import cv2
import numpy as np

def orb_detector(new_image, image_template):
    # Function to compare input image to template and return the number of ORB matches
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template

    # Create ORB detector object
    orb = cv2.ORB_create()
    # Detect keypoints and descriptors in both images
    keypoints_1, descriptors_1 = orb.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image2, None)

    # Check if descriptors exist for both images
    if descriptors_1 is None or descriptors_2 is None:
        return 0

    # Create Brute-Force Matcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Return number of good matches
    good_matches = [m for m in matches if m.distance < 50]
    return len(good_matches)

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Load the reference image (template) in grayscale
image_template = cv2.imread('simpson.png', 0)

threshold = 10  # Set a detection threshold

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for object detection
    height, width = frame.shape[:2]
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # Draw a rectangle to represent the ROI
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)

    # Crop the ROI from the frame
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

    # Flip frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # Calculate the number of ORB matches
    matches = orb_detector(cropped, image_template)

    # Display the number of matches
    cv2.putText(frame, f"Matches: {matches}", (450, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Check if matches exceed the threshold for object detection
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, "Object Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        cv2.waitKey(5000)  # Keep the detection window open for a few seconds
        break  # Exit the loop after detection

    # Show the frame with annotations
    cv2.imshow('Object Detection', frame)

    # Exit if the Enter key (13) is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
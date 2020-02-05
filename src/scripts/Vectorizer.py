import numpy as np
import cv2


config = {
    "gray": False,
    "laplancian": False
}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")

while cap.isOpened():
    # Get frame
    ret, frame = cap.read()
    if ret:
        # Convert to grayscale
        if config["gray"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize
        norm = frame / 255
        # Calculate gradients
        if config["laplancian"]:
            mag = cv2.Laplacian(norm, cv2.CV_64F)
        else:
            gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
            mag, _ = cv2.cartToPolar(gx, gy)
        # Show frame
        cv2.imshow('frame', mag)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

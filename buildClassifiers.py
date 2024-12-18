import numpy as np
import cv2
import glob
import os

# Constants for resizing images
Resize_Image_Width = 28
Resize_Image_Height = 28

# List of character digits (ASCII values for '0' to '9')
char_digits = list(range(48, 58))

# Initialize arrays to hold flattened images and their classifications
flattenedImages = np.empty((0, Resize_Image_Width * Resize_Image_Height))
intClassifications = []

# Folder containing the dataset
dataset_folder = "dataset"

# Loop through each character digit
for i, char in enumerate(char_digits):
    digit_folder = os.path.join(dataset_folder, str(i), "JPEG")

    # Check if the digit folder exists
    if not os.path.exists(digit_folder):
        print(f"Warning: Folder {digit_folder} does not exist.")
        continue

    # Get all JPEG image files in the digit folder
    image_files = glob.glob(os.path.join(digit_folder, '*.jpg'))
    for image_file in image_files:
        imageTraining = cv2.imread(image_file)

        # Check if the image was read successfully
        if imageTraining is None:
            print(f"Error: Unable to read image {image_file}")
            continue

        # Convert the image to grayscale
        imagGray = cv2.cvtColor(imageTraining, cv2.COLOR_BGR2GRAY)
        # Apply binary inverse thresholding
        _, imgThresh = cv2.threshold(imagGray, 150, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded image
        imgContours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area in descending order
        imgContours = sorted(imgContours, key=cv2.contourArea, reverse=True)

        # Process the largest contour
        for contour in imgContours[:1]:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                imgROI = imgThresh[y:y + h, x:x + w]
                # Resize the region of interest to the desired size
                imgResizedRoi = cv2.resize(imgROI, (Resize_Image_Width, Resize_Image_Height))

                # Append the character classification and flattened image
                intClassifications.append(char)
                flattenedImage = imgResizedRoi.reshape((1, Resize_Image_Width * Resize_Image_Height))
                flattenedImages = np.append(flattenedImages, flattenedImage, 0)

# Convert classifications to a numpy array and reshape
fltClassifications = np.array(intClassifications, np.float32)
fltClassifications = fltClassifications.reshape((fltClassifications.size, 1))

print("Training Complete")

# Save classifications and flattened images to text files
np.savetxt("classifications.txt", fltClassifications)
np.savetxt("flatCharImages.txt", flattenedImages)
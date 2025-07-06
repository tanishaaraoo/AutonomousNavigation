import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'ball.jpg'
image = cv2.imread(image_path)

# Display the original image
plt.figure(figsize=(16, 16))
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 2)

# Apply Canny Edge Detection
edge_image = cv2.Canny(blur_image, 50, 150)
plt.subplot(3, 2, 2)
plt.imshow(edge_image, cmap="gray")
plt.title('Canny Edge Detection')

# Apply Hough Circle Transform
circles = cv2.HoughCircles(edge_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=50, maxRadius=100)

# Ensure at least some circles were found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Ensure the circle is within the ball region (you may need to adjust these conditions based on your image)
        if x > 100 and x < 500 and y > 100 and y < 500:
            # Draw the circle on the original image
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Adjust the color and thickness as needed

# Display the result
plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Circle Detected')
plt.show()

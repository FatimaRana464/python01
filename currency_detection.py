import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load pre-trained currency note classification model
model = load_model('currency_note_model.h5')

# Define dictionary to map model predictions to currency denominations
currency_dict = {
    0: '10 Rupees',
    1: '20 Rupees',
    2: '50 Rupees',
    3: '100 Rupees',
    4: '500 Rupees',
    5: '1000 Rupees'
}

# Load image of currency note
image_path = 'currency_note.jpg'
image = cv2.imread(image_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours of the note
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find largest contour (assuming it's the currency note)
contour = max(contours, key=cv2.contourArea)

# Get bounding box of the note
x, y, w, h = cv2.boundingRect(contour)

# Crop the note from the image
cropped_note = image[y:y+h, x:x+w]

# Resize cropped note to match model input size
resized_note = cv2.resize(cropped_note, (224, 224))

# Normalize pixel values to [0, 1]
normalized_note = resized_note / 255.0

# Expand dimensions to match model input shape
input_note = np.expand_dims(normalized_note, axis=0)

# Perform prediction using the pre-trained model
prediction = model.predict(input_note)

# Get predicted currency denomination
predicted_class = np.argmax(prediction)
predicted_currency = currency_dict[predicted_class]

print("Predicted Currency:", predicted_currency)

# Display the original image and cropped note
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cropped_note, cv2.COLOR_BGR2RGB))
plt.title("Cropped Note")

plt.show()

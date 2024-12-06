from keras.api.models import load_model
from keras.src.utils import load_img, img_to_array
import numpy as np

# Load the saved model
loaded_model = load_model('trained_model.keras')

# Make predictions on new data
# For example, if you have a single image 'new_image.jpg' for prediction

# Load the image
new_image = load_img('dataset/valid_set/cat/cat.1102.jpg', target_size=(64, 64))
new_image = img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)

# Normalize the image data
new_image = new_image / 255.0

# Make prediction
prediction = loaded_model.predict(new_image)

# Print the prediction value
print("Prediction value:", prediction)

# Map the prediction to class names
class_names = ['cat', 'dog', 'frog']

# Print the prediction probabilities for each class
for class_name, probability in zip(class_names, prediction[0]):
    print(f"Probability of {class_name}: {probability:.4f}")

predicted_class = class_names[np.argmax(prediction)]
print("Predicted class:", predicted_class)
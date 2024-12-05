from keras.api.models import load_model
from keras.src.utils.image_dataset_utils import image_dataset_from_directory

# Load the saved model
loaded_model = load_model('trained_model.keras')

# Recreate the validation dataset
validation_dataset = image_dataset_from_directory(
    'dataset/valid_set',
    labels='inferred',  # Labels are inferred from subdirectory names
    label_mode='binary',  # Binary classification (cat or dog)
    image_size=(64, 64),  # Ensure image size matches model input
    batch_size=32,  # Adjust batch size if needed
    shuffle=False  # Don't shuffle test data for evaluation
)

# Evaluate the model on the test set and print the accuracy
test_loss, test_acc = loaded_model.evaluate(validation_dataset)
print('Test accuracy:', test_acc)
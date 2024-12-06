import json
import os
import numpy as np
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.layers import BatchNormalization, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, RandomTranslation, RandomCrop
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.api.models import load_model
from keras.src.optimizers import Adam

# Path to the saved model
model_path = 'trained_model.keras'

while True: # Continuous training loop

    # Check if the model file exists
    if os.path.exists(model_path):
        # Load the previously saved model
        classifier = load_model(model_path)
        
        print("Loaded model from disk")
    else:
        print("Creating model from scratch")
        
        classifier = Sequential()
    
        # Add convolutional layers
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu')) # model input size is 64, 64, 3
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(128, (3, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(256, (3, 3), activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
        # Flatten the layers
        classifier.add(Flatten())
    
        # Full connection
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units=256, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units=3, activation='softmax'))
    
        # Compile the CNN
        # Compile the model with gradient clipping
        # optimizer = Adam(clipvalue=1.0)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_dataset = image_dataset_from_directory(
        'dataset/train_set',
        labels='inferred',  # Labels are inferred from subdirectory names
        label_mode='categorical',  # Change to 'categorical' for multi-class classification
        image_size=(64, 64),  # Resize images to 64x64. This is final image size. Ensure image size matches model input
        batch_size=32,  # Adjust batch size if needed
        shuffle=True  # Shuffle train data for evaluation
    )
    
    validation_dataset = image_dataset_from_directory(
       'dataset/valid_set',
        labels='inferred',  # Labels are inferred from subdirectory names
        label_mode='categorical',  # Change to 'categorical' for multi-class classification
        image_size=(64, 64),  # Resize images to 64x64. This is final image size. Ensure image size matches model input
        batch_size=32,  # Adjust batch size if needed
        shuffle=False  # Don't shuffle test data for evaluation
    )
    
    # Normalize the pixel values
    normalization_layer = Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Data augmentation
    data_augmentation = Sequential([
        RandomFlip('horizontal_and_vertical'),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1),
        RandomBrightness(0.1),
        RandomTranslation(0.1, 0.1),
        RandomCrop(64, 64), # Ensure image size matches model input
    ])
    
    # Apply data augmentation to the training dataset
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint('trained_model.keras', monitor='val_loss', save_best_only=True)
    
    # Define a function to get the average of specified metrics from the saved model
    def get_metrics_averages(model_path):
        history_path = model_path + '_history.json'
        if os.path.exists(model_path) and os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            val_accuracy = np.mean(history.get('val_accuracy', []))
            accuracy = np.mean(history.get('accuracy', []))
            val_loss = np.mean(history.get('val_loss', []))
            loss = np.mean(history.get('loss', []))
            
            accuracy_average = np.mean([val_accuracy, accuracy])
            loss_average = np.mean([val_loss, loss])
            
            return val_accuracy, accuracy, val_loss, loss, accuracy_average, loss_average
        return None, None, None, None, float('-inf'), float('inf')
    
    # Ensure the result_logs directory exists
    os.makedirs('result_logs', exist_ok=True)
    
    # Get the averages of the specified metrics from the saved model
    prev_val_accuracy, prev_accuracy, prev_val_loss, prev_loss, best_accuracy_average, best_loss_average =  get_metrics_averages(model_path)
    
    ### Higher values are better for:
    # val_accuracy
    # accuracy
    ### Lower values are better for:
    # val_loss
    # loss
    
    # Train the model
    history = classifier.fit(
        train_dataset,
        epochs=25,
        validation_data=validation_dataset,
        callbacks=[early_stopping, lr_scheduler, model_checkpoint]
    )
    
    # Get the averages of the specified metrics for the current model
    val_accuracy = np.mean(history.history.get('val_accuracy', []))
    accuracy = np.mean(history.history.get('accuracy', []))
    val_loss = np.mean(history.history.get('val_loss', []))
    loss = np.mean(history.history.get('loss', []))
    
    current_accuracy_average = np.mean([val_accuracy, accuracy])
    current_loss_average = np.mean([val_loss, loss])
    
    # Compare the accuracy and loss averages
    accuracy_improvement = current_accuracy_average - best_accuracy_average
    loss_improvement = best_loss_average - current_loss_average
    
    # Log format
    log_format = (
        f"Previous val accuracy: {prev_val_accuracy} - Previous accuracy: {prev_accuracy} - "
        f"Previous val loss: {prev_val_loss} - Previous loss: {prev_loss}\n"
        f"New val accuracy: {val_accuracy} - New accuracy: {accuracy} - "
        f"New val loss: {val_loss} - New loss: {loss}\n\n"
    )
    
    # Save the model only if it shows improvement
    # if accuracy_improvement > 0 and loss_improvement > 0: # yani her ikisi de artmışsa
    if accuracy_improvement + loss_improvement > 0: # yani totalde bir artış varsa, emsela bir tanesi 0.5 ARTMIŞ ve diğeri 0.4 AZALMIŞSA, totalde 0.1 artış olmuş olur ve kaydeder.
        
        classifier.save(model_path)
        
        with open('result_logs/better_results.txt', 'a') as f:
            f.write(log_format)
            
        # Save the training history
        history_path = model_path + '_history.json'
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
    else:
        
        with open('result_logs/same_or_worst_results.txt', 'a') as f:
            f.write(log_format)
<h2>This project implements a Convolutional Neural Network (CNN) for binary image classification (cat or dog). Key features include:</h2>

<p>Data Loading: Automatically loads and labels images from the train_set and valid_set directories.</p>
<p>Image Preprocessing: Resizes images to 64x64 pixels and normalizes pixel values.</p>
<p>Data Augmentation: Applies random flips, rotations, zooms, contrast, brightness adjustments, translations, and cropping to the training dataset to improve model generalization.</p>
<h2>Model Architecture:</h2>
<p>Convolutional Layers: 4 convolutional layers with 32, 64, 128, and 256 filters respectively, each followed by batch normalization and max pooling.</p>
<p>Fully Connected Layers: 2 dense layers with 128 and 256 units respectively, with dropout applied to the first dense layer.</p>
<p>Activation Functions: ReLU activation used in all layers.</p>
<p>Training and Validation: Includes early stopping and learning rate scheduling to optimize training.</p>
<p>Model Persistence: Checks for an existing model file to load from disk or creates a new model from scratch if none exists.</p>

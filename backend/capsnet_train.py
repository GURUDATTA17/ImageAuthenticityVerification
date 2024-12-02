import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import re
import json

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, 
                        message=re.escape("Your `PyDataset` class should call super().__init__(**kwargs) in its constructor."))

# Data parameters
image_size = (128, 128)
batch_size = 32
epochs = 10  # Increased epochs for better training

# Data generator setup (no validation, only training data)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill missing pixels with nearest value
)

train_generator = datagen.flow_from_directory(
    'C:/Users/Admin/Downloads/fake_image/fake_image_detection/real_and_fake_face',  # Change path if necessary
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # Binary classification (real or fake)
)

# Define Squash Layer 
class SquashLayer(layers.Layer): 
    def call(self, inputs):
        norm = tf.reduce_sum(tf.square(inputs), -1, keepdims=True)
        scale = norm / (1 + norm) / tf.sqrt(norm + tf.keras.backend.epsilon())
        return scale * inputs

# Capsule Layer
def CapsuleLayer(inputs, num_capsules, dim_capsules):
    capsules = layers.Dense(num_capsules * dim_capsules)(inputs)
    capsules = layers.Reshape((num_capsules, dim_capsules))(capsules)
    return SquashLayer()(capsules)

# Define the Capsule Network Model
def CapsNetModel(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional layers with Batch Normalization
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten and pass to capsule layer
    x = layers.Flatten()(x)
    capsules = CapsuleLayer(x, num_capsules=16, dim_capsules=8)
    capsules = layers.Flatten()(capsules)
    capsules = layers.Dense(64, activation='relu')(capsules)  # Increased Dense size
    
    # Dropout layer to prevent overfitting
    x = layers.Dropout(0.5)(capsules)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, output)

# Model setup
input_shape = (128, 128, 3)
model = CapsNetModel(input_shape)
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate adjustment
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)

# Model training with callbacks (no validation data)
history = model.fit(
    train_generator,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model in .keras format
model.save('capsnet_model.keras')
print("Model saved as 'capsnet_model.keras'")

# Save training metrics to a file
with open('training_metrics.json', 'w') as f:
    json.dump(history.history, f)
print("Training metrics saved to 'training_metrics.json'")

# Print accuracy for each epoch
for epoch in range(epochs):
    train_acc = history.history['accuracy'][epoch]
    print(f"Epoch {epoch+1}/{epochs} - Training Accuracy: {train_acc:.4f}")

# Check if early stopping stopped the training process
if early_stopping.stopped_epoch > 0:
    print(f"Training stopped early at epoch {early_stopping.stopped_epoch + 1}")

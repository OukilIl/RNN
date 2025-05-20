import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import gc
import kagglehub
import os.path

# Constants
# DATASET_PATH will be set dynamically after download
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CLASSES = 14
BATCH_SIZE = 32
EPOCHS = 20

# Gesture mappings for the game
GESTURE_UP = 2      # HandGesture3 (label 2)
GESTURE_DOWN = 3    # HandGesture4 (label 3)
GESTURE_LEFT = 4    # HandGesture5 (label 4)
GESTURE_RIGHT = 5   # HandGesture6 (label 5)

# Disable mixed precision on CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def clear_memory():
    """Free up memory"""
    gc.collect()
    tf.keras.backend.clear_session()

def load_data(dataset_path):
    """Load and preprocess the dataset"""
    images = []
    labels = []
    
    # The dataset has a specific structure - need to find the HG14 directory
    if "HG14" in os.listdir(dataset_path):
        dataset_path = os.path.join(dataset_path, "HG14")
    if "HG14-Hand Gesture" in os.listdir(dataset_path):
        dataset_path = os.path.join(dataset_path, "HG14-Hand Gesture")
    
    gesture_folders = sorted(os.listdir(dataset_path))
    print(f"Found gesture folders: {gesture_folders}")
    
    # Use 500 samples per class for faster training
    max_samples_per_class = 500
    
    for gesture_folder_name in gesture_folders:
        if not gesture_folder_name.startswith("Gesture_"):
            print(f"Skipping non-gesture folder: {gesture_folder_name}")
            continue

        try:
            label = int(gesture_folder_name.split('_')[1]) - 1  # Labels start from 0
        except (IndexError, ValueError):
            print(f"Warning: Could not parse label from folder name {gesture_folder_name}. Skipping.")
            continue
            
        gesture_folder_path = os.path.join(dataset_path, gesture_folder_name)
        if not os.path.isdir(gesture_folder_path):
            print(f"Skipping non-directory: {gesture_folder_path}")
            continue

        print(f"Processing folder: {gesture_folder_name} (Label: {label})")
        image_count = 0
        
        for file_name in sorted(os.listdir(gesture_folder_path))[:max_samples_per_class]:
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                img_path = os.path.join(gesture_folder_path, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Preprocess image
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = img / 255.0  # Normalize
                
                images.append(img)
                labels.append(label)
                image_count += 1
                
                # Clear memory periodically
                if len(images) % 1000 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing image {file_name}: {e}")
                continue
                
        print(f"Loaded {image_count} images from {gesture_folder_name}")
            
    if not images or not labels:
        print("Error: No images or labels were loaded. Aborting.")
        return None, None

    print(f"Total images loaded: {len(images)}")
    images = np.array(images)
    labels = np.array(labels)
    
    # One-hot encode labels
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    
    return images, labels

def create_custom_model():
    """Create a custom CNN model for hand gesture recognition"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Classification layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with data augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Callbacks for training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            filepath='best_hand_gesture_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create generator
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False  # Better for CPU training
    )
    
    return history

def plot_training_history(history):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as training_history.png")

if __name__ == '__main__':
    clear_memory()
    
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("gulerosman/hg14-handgesture14-dataset")
    print("Path to dataset files:", dataset_path)
    
    print("Loading dataset...")
    images, labels = load_data(dataset_path)
    
    if images is None or labels is None:
        print("Failed to load data. Exiting.")
        exit()
        
    print(f"Data loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Free up memory
    images = labels = None
    clear_memory()
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    print("Creating custom model...")
    model = create_custom_model()
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")
    
    print("Saving model...")
    model.save("hand_gesture_model.h5")
    print("Model saved as hand_gesture_model.h5")
    
    # Plot training history
    plot_training_history(history)
    
    print("Training complete!") 
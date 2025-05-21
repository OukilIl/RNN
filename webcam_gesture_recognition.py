import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import time

class HandGestureRecognizer:
    def __init__(self, model_path, img_size=(128, 128)):
        # Load the trained model
        self.model = load_model(model_path)
        self.img_size = img_size
        
        # Class labels (according to the dataset)
        self.labels = [
            "Gesture_0", "Gesture_1", "Gesture_2", "Gesture_3", 
            "Gesture_4", "Gesture_5", "Gesture_6", "Gesture_7", 
            "Gesture_8", "Gesture_9", "Gesture_10", "Gesture_11", 
            "Gesture_12", "Gesture_13"
        ]
        
        # Game control mappings
        self.control_map = {
            "Gesture_5": "UP",     # index 1
            "Gesture_13": "DOWN",   # index 2
            "Gesture_6": "LEFT",   # index 3
            "Gesture_2": "RIGHT",  # index 4
        }
        
        # Initialize prediction history for smoothing
        self.history = []
        self.max_history = 5
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.cooldown = 0.5  # seconds between predictions
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=50, detectShadows=False)
        self.bg_initialized = False
        self.frames_for_bg = 30  # Number of frames to build background model
        self.frame_count = 0
        
        # Parameters for skin color detection
        self.use_skin_detection = True
        self.min_YCrCb = np.array([0, 135, 85], np.uint8)
        self.max_YCrCb = np.array([255, 180, 135], np.uint8)
        
    def remove_background(self, frame):
        """Remove background from input frame"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # If we're still initializing the background model
        if not self.bg_initialized:
            self.frame_count += 1
            if self.frame_count >= self.frames_for_bg:
                self.bg_initialized = True
                print("Background model initialized")
            return frame  # Return original frame during initialization
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Create a copy of the original frame
        processed_frame = frame.copy()
        
        # Apply skin color detection if enabled
        if self.use_skin_detection:
            # Convert to YCrCb color space 
            frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # Find skin pixels
            skin_mask = cv2.inRange(frame_YCrCb, self.min_YCrCb, self.max_YCrCb)
            
            # Combine skin mask with foreground mask for better hand detection
            combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
            
            # Apply morphological operations to clean up skin mask
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Apply combined mask
            processed_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
        else:
            # Apply foreground mask only
            processed_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        # Debug - show mask in corner (uncomment if needed)
        # h, w = frame.shape[:2]
        # frame[h-100:h, w-100:w] = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), (100, 100))
        
        return processed_frame
        
    def preprocess_frame(self, frame):
        """Preprocess a video frame for prediction"""
        # Remove background
        processed_frame = self.remove_background(frame)
        
        # Resize to model input size
        img = cv2.resize(processed_frame, self.img_size)
        
        # Convert to RGB (model was trained on RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame):
        """Make a prediction on the current frame"""
        # Preprocess the frame
        preprocessed = self.preprocess_frame(frame)
        
        # Get prediction
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        # Get the predicted class index and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Add to history for smoothing
        self.history.append(predicted_class_idx)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Get most common prediction from history
        from collections import Counter
        most_common = Counter(self.history).most_common(1)[0][0]
        
        # Get the label and game control (if applicable)
        label = self.labels[most_common]
        game_control = self.control_map.get(label, None)
        
        # Limit prediction frequency to reduce flickering
        current_time = time.time()
        if current_time - self.last_prediction_time > self.cooldown:
            self.last_prediction = game_control
            self.last_prediction_time = current_time
        
        return label, confidence, self.last_prediction

    def visualize_prediction(self, frame, label, confidence, game_control):
        """Add visualization to the frame"""
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Process the frame to show the segmentation
        processed_frame = self.remove_background(frame)
        
        # Create a composite image - original on left, processed on right
        h, w = frame.shape[:2]
        composite = np.zeros((h, w*2, 3), dtype=np.uint8)
        composite[:, :w] = frame
        composite[:, w:] = processed_frame
        
        # Draw a line between the two frames
        cv2.line(composite, (w, 0), (w, h), (255, 255, 255), 1)
        
        # Add text for the predicted label and confidence
        cv2.putText(
            composite, 
            f"{label} ({confidence:.2f})", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        # Add text for the game control (if applicable)
        if game_control:
            cv2.putText(
                composite, 
                f"Game Control: {game_control}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
        
        # Add text labels for the frames
        cv2.putText(
            composite,
            "Original",
            (w//2 - 50, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            composite,
            "Processed",
            (w + w//2 - 60, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return composite
    
    def run_webcam(self):
        """Run the hand gesture recognition with webcam input"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Webcam started. Press 'q' to quit.")
        print("Initializing background model, please keep the background clear...")
        
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Make a prediction
            label, confidence, game_control = self.predict(frame)
            
            # Visualize the prediction
            vis_frame = self.visualize_prediction(frame, label, confidence, game_control)
            
            # Display the frame
            cv2.imshow("Hand Gesture Recognition", vis_frame)
            
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Toggle skin detection
                self.use_skin_detection = not self.use_skin_detection
                print(f"Skin detection: {'ON' if self.use_skin_detection else 'OFF'}")
            elif key == ord('b'):
                # Reset background model
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=50, detectShadows=False)
                self.bg_initialized = False
                self.frame_count = 0
                print("Resetting background model...")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    def get_current_control(self):
        """Return the current game control (for use by game)"""
        return self.last_prediction

if __name__ == "__main__":
    try:
        # Initialize the hand gesture recognizer
        recognizer = HandGestureRecognizer("hand_gesture_model.h5")
        
        # Run the webcam capture and prediction
        recognizer.run_webcam()
    except Exception as e:
        print(f"Error: {e}") 
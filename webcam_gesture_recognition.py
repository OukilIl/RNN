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
            "Gesture_1", "Gesture_2", "Gesture_3", "Gesture_4", 
            "Gesture_5", "Gesture_6", "Gesture_7", "Gesture_8", 
            "Gesture_9", "Gesture_10", "Gesture_11", "Gesture_12", 
            "Gesture_13", "Gesture_14"
        ]
        
        # Game control mappings
        self.control_map = {
            "Gesture_3": "UP",     # index 2
            "Gesture_4": "DOWN",   # index 3
            "Gesture_5": "LEFT",   # index 4
            "Gesture_6": "RIGHT",  # index 5
        }
        
        # Initialize prediction history for smoothing
        self.history = []
        self.max_history = 5
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.cooldown = 0.5  # seconds between predictions
        
    def preprocess_frame(self, frame):
        """Preprocess a video frame for prediction"""
        # Resize to model input size
        img = cv2.resize(frame, self.img_size)
        
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
        
        # Add text for the predicted label and confidence
        cv2.putText(
            vis_frame, 
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
                vis_frame, 
                f"Game Control: {game_control}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
        
        return vis_frame
    
    def run_webcam(self):
        """Run the hand gesture recognition with webcam input"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Webcam started. Press 'q' to quit.")
        
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
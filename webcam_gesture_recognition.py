import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import time

# Constants
CONFIDENCE_THRESHOLD = 0.7  # Only register gestures with at least 70% confidence

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
        
        # Initialize background subtractor with less aggressive parameters
        # Reduced history, lower threshold for more sensitivity
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        self.bg_initialized = False
        self.frames_for_bg = 30  # Number of frames to build background model
        self.frame_count = 0
        
        # Parameters for skin color detection - wider range for better detection
        self.use_skin_detection = True
        
        # HSV color space parameters for skin detection (provides better results for a variety of skin tones)
        self.use_hsv = True
        self.min_HSV = np.array([0, 20, 70], np.uint8)
        self.max_HSV = np.array([25, 255, 255], np.uint8)
        
        # YCrCb parameters (fallback)
        self.min_YCrCb = np.array([0, 130, 75], np.uint8)
        self.max_YCrCb = np.array([255, 185, 140], np.uint8)
        
        # Contour filtering settings
        self.min_contour_area = 3000  # Minimum area to consider as a hand
        
    def remove_background(self, frame):
        """Remove background from input frame with improved hand detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # If we're still initializing the background model
        if not self.bg_initialized:
            self.frame_count += 1
            if self.frame_count >= self.frames_for_bg:
                self.bg_initialized = True
                print("Background model initialized")
            return frame  # Return original frame during initialization
        
        # Apply morphological operations to clean up mask - gentler to preserve detail
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Create a copy of the original frame
        processed_frame = frame.copy()
        
        # Apply skin color detection if enabled
        if self.use_skin_detection:
            if self.use_hsv:
                # Convert to HSV color space for better skin tone range detection
                frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                skin_mask = cv2.inRange(frame_HSV, self.min_HSV, self.max_HSV)
                
                # Second range for handling red hues (HSV wraps around for red)
                lower_red = np.array([170, 20, 70], np.uint8)
                upper_red = np.array([180, 255, 255], np.uint8)
                skin_mask2 = cv2.inRange(frame_HSV, lower_red, upper_red)
                
                # Combine the two skin masks
                skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
            else:
                # Use YCrCb as fallback
                frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                skin_mask = cv2.inRange(frame_YCrCb, self.min_YCrCb, self.max_YCrCb)
            
            # Apply gentle morphological operations to clean up skin mask
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Create a combined mask:
            # 1. More weight to skin detection for focused hand detection
            # 2. Less weight to background subtraction to avoid losing hand details
            combined_mask = cv2.bitwise_or(skin_mask, fg_mask)
            
            # Apply the combined mask to the original frame
            processed_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
            
            # Find contours to isolate the hand
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If contours are found, focus on the largest one (likely the hand)
            if contours:
                # Sort contours by area (descending)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Filter out small contours that are unlikely to be a hand
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
                
                if large_contours:
                    # Create a mask with only the largest contour(s)
                    hand_mask = np.zeros_like(combined_mask)
                    cv2.drawContours(hand_mask, [large_contours[0]], -1, 255, -1)
                    
                    # If multiple large contours are close together, they might be part of the same hand
                    if len(large_contours) > 1:
                        for cnt in large_contours[1:3]:  # Consider next 2 largest contours
                            # Get the distance between contours
                            M1 = cv2.moments(large_contours[0])
                            if M1['m00'] != 0:
                                cx1 = int(M1['m10']/M1['m00'])
                                cy1 = int(M1['m01']/M1['m00'])
                                
                                M2 = cv2.moments(cnt)
                                if M2['m00'] != 0:
                                    cx2 = int(M2['m10']/M2['m00'])
                                    cy2 = int(M2['m01']/M2['m00'])
                                    
                                    # If centers are close, add to the mask
                                    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                                    if dist < 100:  # Threshold for closeness
                                        cv2.drawContours(hand_mask, [cnt], -1, 255, -1)
                    
                    # Dilate the hand mask slightly to ensure we get the full hand
                    hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
                    
                    # Apply the hand mask to the original frame
                    processed_frame = cv2.bitwise_and(frame, frame, mask=hand_mask)
                    
                    # Optional: Draw a rectangle around the hand contour
                    # x, y, w, h = cv2.boundingRect(large_contours[0])
                    # cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Apply foreground mask only if skin detection is disabled
            processed_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        # Debug - show mask in corner (uncomment if needed)
        h, w = frame.shape[:2]
        small_mask = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), (80, 80))
        processed_frame[h-80:h, w-80:w] = small_mask
        
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
        
        # Get the label
        label = self.labels[most_common]
        
        # Get the game control (if applicable and confidence is above threshold)
        game_control = None
        if confidence >= CONFIDENCE_THRESHOLD:
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
            (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255), 
            2
        )
        
        # Add text for confidence threshold
        cv2.putText(
            composite,
            f"Threshold: {CONFIDENCE_THRESHOLD:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Add text for the game control (if applicable)
        if game_control:
            cv2.putText(
                composite, 
                f"Game Control: {game_control}", 
                (10, 90), 
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
        print("Press 's' to toggle skin detection method (HSV/YCrCb)")
        print("Press 'b' to reset background model")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f} - gestures below this won't trigger controls")
        
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
                # Toggle skin detection method
                self.use_hsv = not self.use_hsv
                print(f"Using {'HSV' if self.use_hsv else 'YCrCb'} color space for skin detection")
            elif key == ord('b'):
                # Reset background model
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
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
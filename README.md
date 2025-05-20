# Hand Gesture Controlled Sokoban Game

This project implements a Sokoban game that can be controlled using hand gestures through a webcam. It uses a custom Convolutional Neural Network (CNN) model trained on the HG14 hand gesture dataset to recognize different hand poses in real-time.

## Project Files

- `train_gesture_model.py` - Script to train the custom CNN model on the HG14 dataset
- `webcam_gesture_recognition.py` - Module for capturing webcam input and recognizing hand gestures
- `sokoban_game.py` - Implementation of the Sokoban puzzle game using Pygame
- `gesture_controlled_sokoban.py` - Main script that integrates the gesture recognition with the game

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pygame
- scikit-learn

You can install the required dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib pygame scikit-learn
```

## Dataset

This project uses the HG14 (HandGesture14) dataset which contains 14 different hand gesture classes. The dataset should be already downloaded at:

```
C:/Users/oukil/.cache/kagglehub/datasets/gulerosman/hg14-handgesture14-dataset/versions/1/HG14/HG14-Hand Gesture
```

## Usage

### 1. Train the Model

First, train the hand gesture recognition model:

```bash
python train_gesture_model.py
```

This will:
- Load and preprocess the HG14 dataset
- Train a custom CNN model 
- Save the trained model as `hand_gesture_model.h5`

The training process may take some time depending on your hardware. For CPUs, the training parameters have been optimized for better performance.

### 2. Run the Game

Once the model is trained, you can run the game with:

```bash
python gesture_controlled_sokoban.py
```

### Controls

The game can be controlled using both keyboard and hand gestures:

**Keyboard Controls:**
- Arrow keys: Move the player
- R: Reset level
- N: Next level (when completed)
- ESC: Quit

**Hand Gesture Controls:**
- Gesture 3: Move Up
- Gesture 4: Move Down
- Gesture 5: Move Left
- Gesture 6: Move Right

## Game Rules

Sokoban (倉庫番) is a classic puzzle game where the player pushes boxes onto target locations:

1. The player can move in four directions (up, down, left, right)
2. Boxes can be pushed (but not pulled)
3. Only one box can be pushed at a time
4. The level is completed when all boxes are on target locations

## Modifying the Game

You can modify the levels in `sokoban_game.py` by editing the `LEVELS` list. Each level is represented as a list of strings, where:
- `#` represents a wall
- ` ` (space) represents an empty floor
- `P` represents the player
- `B` represents a box
- `T` represents a target
- `X` represents a box on a target
- `Y` represents a player on a target

## Acknowledgments

This project uses the HG14 hand gesture dataset as referenced in the paper "Hand Gesture Recognition with Two Stage Approach Using Transfer Learning and Deep Ensemble Learning" by Serkan Savaşa and Atilla Ergüzena. 
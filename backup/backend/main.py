import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

# Load the pre-trained model
model_path = resource_path('modelislv17.p')
with open(model_path, 'rb') as file:
    model_dict = pickle.load(file)
model = model_dict['model']

# Variables for tracking predictions
prev_character = None
consecutive_count = 0
required_count = 10
current_word = []
sentence = []

def process_frame(frame_data):
    # Decode base64 image
    img_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    H, W, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    pose_results = pose.process(img_rgb)
    hands_results = hands.process(img_rgb)
    
    # Initialize empty lists
    pose_data, lh, rh, x_all, y_all = [], [], [], [], []
    
    # Extract pose keypoints
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            x_all.append(lm.x)
            y_all.append(lm.y)
            pose_data.extend([lm.x, lm.y, lm.visibility])
    
    # Extract hand keypoints
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            hand_label = hands_results.multi_handedness[idx].classification[0].label
            for lm in hand_landmarks.landmark:
                x_all.append(lm.x)
                y_all.append(lm.y)
                if hand_label == "Left":
                    lh.extend([lm.x, lm.y])
                elif hand_label == "Right":
                    rh.extend([lm.x, lm.y])
    
    # Process data if landmarks were detected
    if x_all and y_all:
        # Normalize coordinates
        x_min, y_min = min(x_all), min(y_all)
        pose_data = [(val - x_min if i % 3 == 0 else val - y_min if i % 3 == 1 else val) 
                    for i, val in enumerate(pose_data)]
        lh = [(val - x_min if i % 2 == 0 else val - y_min) for i, val in enumerate(lh)]
        rh = [(val - x_min if i % 2 == 0 else val - y_min) for i, val in enumerate(rh)]
        
        # Pad or truncate data
        lh = lh[:42] + [0] * max(0, 42 - len(lh))
        rh = rh[:42] + [0] * max(0, 42 - len(rh))
        pose_data = pose_data[:99] + [0] * max(0, 99 - len(pose_data))
        
        # Prepare data for prediction
        data_aux = pose_data + lh + rh
        prediction = model.predict([np.asarray(data_aux)])
        
        return {
            'prediction': prediction[0],
            'bounds': {
                'x1': int(min(x_all) * W) - 10,
                'y1': int(min(y_all) * H) - 10,
                'x2': int(max(x_all) * W) + 10,
                'y2': int(max(y_all) * H) + 10
            }
        }
    return None

@app.websocket("/ws/full")
async def websocket_endpoint(websocket: WebSocket):
    global prev_character, consecutive_count, current_word, sentence
    
    await websocket.accept()
    try:
        while True:
            # Receive base64 encoded frame
            frame_data = await websocket.receive_text()
            
            # Process the frame
            result = process_frame(frame_data)
            
            if result:
                predicted_character = result['prediction']
                
                # Handle consecutive detections
                if predicted_character == prev_character:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    prev_character = predicted_character
                
                # Register character after confirmation
                if consecutive_count == required_count:
                    consecutive_count = 0
                    if predicted_character != 'null':
                        current_word.append(predicted_character)
                
                # Send back the results
                await websocket.send_json({
                    'prediction': predicted_character,
                    'bounds': result['bounds'],
                    'current_word': ''.join(current_word),
                    'sentence': ' '.join(sentence)
                })
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

def start_server():
    """Function to start the server"""
    uvicorn.run(app, host="0.0.0.0", port=9002)

if __name__ == "__main__":
    start_server()
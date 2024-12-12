import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import base64
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

# Load the pre-trained models
with open('./modelislv17.p', 'rb') as file:
    model_dict_full = pickle.load(file)
model_full = model_dict_full['model']

with open('./modelislv10.p', 'rb') as file:
    model_dict_hands = pickle.load(file)
model_hands = model_dict_hands['model']

# Variables for tracking predictions
prev_character = None
consecutive_count = 0
required_count = 10
current_word = []
sentence = []

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()
manager1 = ConnectionManager()

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
        
        # Prepare data for both models
        data_aux_full = pose_data + lh + rh
        data_aux_hands = lh + rh
        
        # Get predictions from both models
        prediction_full = model_full.predict([np.asarray(data_aux_full)])
        prediction_hands = model_hands.predict([np.asarray(data_aux_hands)])
        
        return {
            'prediction_full': prediction_full[0],
            'prediction_hands': prediction_hands[0],
            'bounds': {
                'x1': int(min(x_all) * W) - 10,
                'y1': int(min(y_all) * H) - 10,
                'x2': int(max(x_all) * W) + 10,
                'y2': int(max(y_all) * H) + 10
            }
        }
    return None

@app.websocket("/ws/full")
async def websocket_endpoint_full(websocket: WebSocket):
    global prev_character, consecutive_count, current_word, sentence
    
    await manager.connect(websocket)
    try:
        while True:
            # Receive base64 encoded frame
            frame_data = await websocket.receive_text()
            
            # Process the frame
            result = process_frame(frame_data)
            
            if result:
                predicted_character = result['prediction_full']
                
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
                await manager.send_message({
                    'prediction': predicted_character,
                    'bounds': result['bounds'],
                    'current_word': ''.join(current_word),
                    'sentence': ' '.join(sentence)
                }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/hands")
async def websocket_endpoint_hands(websocket: WebSocket):
    await manager1.connect(websocket)
    try:
        while True:
            frame_data = await websocket.receive_text()
            result = process_frame(frame_data)
            
            if result:
                await manager1.send_message({
                    'prediction': result['prediction_hands'],
                    'bounds': result['bounds']
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9002)
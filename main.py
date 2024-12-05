import sklearn
import cv2
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import sys
import numpy as np
import base64
import pickle
import asyncio
from typing import Dict, List

app = FastAPI()
active_connections: Dict[int, WebSocket] = {}

def get_resource_path(filename):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return os.path.join(os.path.dirname(__file__), filename)
    
def convert_landmarks_to_dict(hand_landmarks) -> List[dict]:
    landmarks_list = []
    for landmark in hand_landmarks.landmark:
        landmarks_list.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    return landmarks_list

def process_base64_image(base64_string):
    try:
        # Remove header more efficiently
        split_point = base64_string.find('base64,')
        if split_point != -1:
            base64_string = base64_string[split_point + 7:]
        
        # Use numpy's built-in base64 decoder for better performance
        img_data = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        return img
        
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        return None

def extract_landmarks(hand_landmarks):
    """Optimized landmark extraction"""
    data_aux = []
    x_ = []
    y_ = []
    
    # Pre-allocate arrays
    landmarks = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
    x_ = landmarks[:, 0]
    y_ = landmarks[:, 1]
    
    # Vectorized normalization
    x_min, y_min = np.min(x_), np.min(y_)
    normalized = np.column_stack((x_ - x_min, y_ - y_min))
    return normalized.flatten()

@app.websocket("/asl")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_connections[connection_id] = websocket
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        model_path = get_resource_path("model1.p")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)['model']

        while True:
            if connection_id not in active_connections:
                break

            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except (asyncio.TimeoutError, WebSocketDisconnect):
                continue

            frame = process_base64_image(data)
            if frame is None:
                continue

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                if connection_id in active_connections:
                    await websocket.send_json({"message": "No hands detected"})
                continue

            # Process first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = extract_landmarks(hand_landmarks)
            
            prediction = model.predict(data_aux.reshape(1, -1))
            
            if connection_id in active_connections:
                await websocket.send_json({
                    "multiHandLandmarks": [convert_landmarks_to_dict(hand_landmarks)],
                    "resultData": prediction[0]
                })

    except Exception as e:
        print(f"Error: {str(e)}")
        if connection_id in active_connections:
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]
        hands.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=9000)
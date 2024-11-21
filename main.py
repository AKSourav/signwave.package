from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import sys
import numpy as np
import base64
import mediapipe as mp
import pickle
import asyncio
from typing import Dict, List
import cv2

app = FastAPI()

# Store active connections
active_connections: Dict[int, WebSocket] = {}

# Convert landmarks to serializable format
# Function to convert MediaPipe landmarks to serializable format
def convert_landmarks_to_dict(hand_landmarks) -> List[dict]:
    landmarks_list = []
    for landmark in hand_landmarks.landmark:
        landmarks_list.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    return landmarks_list

def get_resource_path(filename):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return os.path.join(os.path.dirname(__file__), filename)

def process_base64_image(base64_string):
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Remove any whitespace
        base64_string = base64_string.strip()
        
        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        # img = None
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        return img
        
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        return None

@app.websocket("/asl")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Add to active connections
    connection_id = id(websocket)
    active_connections[connection_id] = websocket
    
    # Initialize MediaPipe Hands for this connection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        # Load the model
        model_path = get_resource_path("model1.p")
        with open(model_path, 'rb') as file:
            model_dict = pickle.load(file)
        model = model_dict['model']

        while True:
            try:
                # Check if connection is still active
                if connection_id not in active_connections:
                    break

                # Receive the frame with a timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break
                
                # Process the image
                frame = process_base64_image(data)
                if frame is None:
                    continue

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe
                results = hands.process(frame_rgb)
                
                if not results.multi_hand_landmarks:
                    if connection_id in active_connections:
                        await websocket.send_json({"message": "No hands detected"})
                    continue

                # Process landmarks for model
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                
                serialized_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    serialized_landmarks.append(convert_landmarks_to_dict(hand_landmarks))
                
                # Send results if still connected
                if connection_id in active_connections:
                    await websocket.send_json({
                        "multiHandLandmarks": serialized_landmarks,
                        "resultData": prediction[0]
                    })

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                if connection_id in active_connections:
                    try:
                        await websocket.send_json({"error": str(e)})
                    except:
                        break
                continue

    except Exception as e:
        print(f"Error: {str(e)}")
        if connection_id in active_connections:
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
    finally:
        # Cleanup
        if connection_id in active_connections:
            del active_connections[connection_id]
        
        # Properly close MediaPipe hands
        try:
            hands.close()
        except:
            pass

if __name__ == '__main__':
    import uvicorn
    import webbrowser
    webbrowser.open("http://127.0.0.1:3000")
    uvicorn.run(app=app, host='0.0.0.0', port=9000)
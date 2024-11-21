import os
import sys
import pickle
import json
import base64
import cv2
import numpy as np
from flask import Flask
from flask_sock import Sock
import mediapipe as mp
from typing import List

app = Flask(__name__)
sockets = Sock(app)

# Store active connections
active_connections = {}

# Convert landmarks to serializable format
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
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        base64_string = base64_string.strip()
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        return None

@sockets.route('/asl')
def websocket_endpoint(ws):
    connection_id = id(ws)
    active_connections[connection_id] = ws

    # Initialize MediaPipe Hands for this connection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Load the model once when the server starts
    model_path = get_resource_path("model1.p")
    try:
        with open(model_path, 'rb') as file:
            model_dict = pickle.load(file)
        model = model_dict['model']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

    if model is None:
        ws.send(json.dumps({"error": "Model loading failed."}))
        return

    try:
        while True:
            try:
                # Receive the frame with a timeout
                data = ws.receive()
                if data is None:
                    continue
                
                # Process the image
                frame = process_base64_image(data)
                if frame is None:
                    continue

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe
                results = hands.process(frame_rgb)
                
                if not results.multi_hand_landmarks:
                    ws.send(json.dumps({"message": "No hands detected"}))
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
                ws.send(json.dumps({
                    "multiHandLandmarks": serialized_landmarks,
                    "resultData": prediction[0].tolist()  # Convert numpy array to list
                }))
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                try:
                    ws.send(json.dumps({"error": str(e)}))
                except Exception as send_error:
                    print(f"Error sending data to client: {send_error}")
                break

    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            ws.send(json.dumps({"error": str(e)}))
        except:
            pass
    finally:
        # Cleanup
        if connection_id in active_connections:
            del active_connections[connection_id]
        
        # Properly close MediaPipe hands
        try:
            hands.close()
        except Exception as close_error:
            print(f"Error closing MediaPipe: {close_error}")

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:3000")
    app.run(host='0.0.0.0', port=9000)

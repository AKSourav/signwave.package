# Sign Language Recognition Application Documentation

## Overview
This application is a real-time sign language recognition system that uses computer vision and machine learning to detect and interpret sign language gestures through a webcam feed. The system processes video frames, extracts pose and hand landmarks, and predicts sign language characters.

## Technical Architecture

### Components
- **FastAPI Backend Server**: Handles WebSocket connections and real-time processing
- **MediaPipe Integration**: Extracts pose and hand landmarks
- **Machine Learning Model**: Processes landmarks to predict sign language characters
- **WebSocket Protocol**: Enables real-time communication between client and server

### Dependencies
```python
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
```

## Features

### Core Functionality
1. Real-time video processing
2. Hand and pose landmark detection
3. Sign language character prediction
4. Continuous gesture tracking
5. Word and sentence formation
6. WebSocket-based real-time communication

### Model Integration
- Pre-trained model loaded from `modelislv17.p`
- Resource path handling for both development and production environments
- Pickle-based model serialization

## API Endpoints

### WebSocket Endpoint
```
WebSocket URL: ws://localhost:9002/ws/full
```

#### Response Format
```json
{
    "prediction": "character",
    "bounds": {
        "x1": int,
        "y1": int,
        "x2": int,
        "y2": int
    },
    "current_word": "string",
    "sentence": "string"
}
```

## Implementation Details

### Frame Processing Pipeline
1. **Image Decoding**
   - Base64 decode incoming frame
   - Convert to OpenCV format
   - Transform to RGB color space

2. **Landmark Detection**
   - Extract pose landmarks using MediaPipe Pose
   - Extract hand landmarks using MediaPipe Hands
   - Process both left and right hand data

3. **Data Preprocessing**
   - Normalize coordinates
   - Pad or truncate data to fixed lengths
   - Combine pose and hand data for prediction

4. **Prediction Processing**
   - Track consecutive predictions
   - Implement confirmation threshold (10 consecutive matches)
   - Build words and sentences from confirmed predictions

### Configuration

#### CORS Settings
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

#### MediaPipe Configuration
```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.3
)
```

## Running the Application

### Development Mode
```bash
python main.py
```

The server will start on port 9002 and listen for WebSocket connections.

### Production Deployment
1. Ensure all dependencies are installed
2. Configure appropriate CORS settings
3. Start the server using a production-grade ASGI server

## Error Handling
- WebSocket connection management
- Frame processing error handling
- Exception logging and graceful connection closure

## Data Structures

### Tracking Variables
```python
prev_character = None
consecutive_count = 0
required_count = 10
current_word = []
sentence = []
```

### Landmark Data Format
- Pose Data: 99 values (33 landmarks × 3 coordinates)
- Hand Data: 42 values per hand (21 landmarks × 2 coordinates)

## Best Practices
1. Use resource_path for file handling
2. Implement proper error handling
3. Clean WebSocket connection management
4. Efficient frame processing
5. Careful memory management
6. Proper data normalization

## Future Improvements
1. Add authentication
2. Implement rate limiting
3. Add data validation
4. Improve error handling
5. Add performance monitoring
6. Implement caching mechanisms

## Performance Considerations
- Frame processing optimization
- Memory usage management
- Connection handling efficiency
- Model inference optimization

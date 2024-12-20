# Real-time Emotion Detection 😊 😢 😠

A real-time facial emotion detection system using OpenCV and DeepFace. This application captures video from your webcam and analyzes facial expressions to detect emotions in real-time.

## ✨ Features

- Real-time face detection using Haar Cascade Classifier
- Emotion analysis using DeepFace
- Live webcam feed processing
- Visual feedback with bounding boxes
- Real-time emotion display
- Support for multiple face detection

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install opencv-python
pip install deepface
pip install numpy
```

### Running the Application

1. Clone this repository
2. Navigate to the project directory
3. Run the script:
```bash
python main.py
```

## 💡 How It Works

The application works in three main steps:

1. **Face Detection**
   - Uses Haar Cascade Classifier for face detection
   - Draws green rectangles around detected faces

2. **Emotion Analysis**
   - Processes each frame using DeepFace
   - Analyzes facial expressions in real-time
   - Determines the dominant emotion

3. **Display**
   - Shows live webcam feed
   - Overlays emotion labels
   - Displays face detection boundaries

## 🎮 Controls

- Press 'q' to quit the application
- The webcam window will show:
  - Green boxes around detected faces
  - Current dominant emotion displayed in red text

## 🛠️ Technical Details

### Key Components

```python
# Face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion analysis
result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
```

### Detectable Emotions

- Happy
- Sad
- Angry
- Neutral
- Surprised
- Fearful
- Disgusted

## ⚙️ Customization

You can modify:
- Detection sensitivity
- Display parameters
- Emotion text size and color
- Face detection parameters

Example:
```python
# Adjust face detection parameters
faces = faceCascade.detectMultiScale(gray, 
    scaleFactor=1.1,  # Adjust this value
    minNeighbors=4    # Adjust this value
)

# Modify text display
font_scale = 3        # Change text size
font_color = (0,0,255) # Change text color (BGR)
```

## 🔍 Troubleshooting

1. **No Webcam Detection**
   - Check webcam connections
   - Verify webcam permissions
   - Try alternative video capture device:
     ```python
     cap = cv2.VideoCapture(1)  # Try different device numbers
     ```

2. **Poor Face Detection**
   - Ensure good lighting
   - Face the camera directly
   - Adjust face detection parameters

3. **Performance Issues**
   - Check system resources
   - Reduce frame resolution if needed
   - Optimize detection parameters

## 🔧 Performance Tips

- Use good lighting conditions
- Maintain appropriate distance from camera
- Ensure face is clearly visible
- Avoid rapid movements
- Keep face within frame boundaries

## 🚨 System Requirements

- Python 3.6+
- Webcam or video input device
- Sufficient CPU for real-time processing
- Minimum 4GB RAM recommended

## 📝 Notes

- The application processes frames in real-time
- Emotion detection accuracy depends on:
  - Lighting conditions
  - Face visibility
  - Camera quality
  - Face angle

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Multiple face emotion tracking
- Emotion history tracking
- Additional visualization options
- Performance optimizations
- Support for video file input

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- OpenCV team
- DeepFace developers
- Haar Cascade Classifier developers

## ⚠️ Limitations

- Best performance with front-facing faces
- Requires good lighting conditions
- May have reduced accuracy with masks
- Processing speed depends on hardware

## 📞 Support

For issues and feature requests, please create an issue in the repository.
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

model = load_model('model.h5', compile=False)

# Dictionary mapping emotion index to label
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the MTCNN detector
detector = MTCNN()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face to match the model input size
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi.reshape(1, 48, 48, 1)

        # Predict the emotion
        result = model.predict(face_roi)
        result = list(result[0])
        idx = result.index(max(result))

        # Get emotion label
        predicted_label = label_dict[idx]
       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

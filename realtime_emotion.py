import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Load emotion detection model
with open("facialemotionmodel.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("facialemotionmodel.weights.h5")
print("[INFO] Loaded model from disk.")

# Emotion labels (adjust if order differs)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time emotion detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = grayscale[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-time Emotion Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

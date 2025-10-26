import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle


MODEL_FILE = "asl_landmark_model.keras"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_FILE)
with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)
with open(ENCODER_FILE, "rb") as f:
    label_encoder = pickle.load(f)

print("✅ Loaded model and preprocessors")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks_scaled = scaler.transform([landmarks])
            pred = model.predict(landmarks_scaled, verbose=0)
            class_idx = np.argmax(pred)
            letter = label_encoder.inverse_transform([class_idx])[0]
            confidence = np.max(pred)

            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f"{letter} ({confidence*100:.1f}%)", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🖐️ Session ended.")

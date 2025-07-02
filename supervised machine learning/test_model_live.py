import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("my_digit_model.keras")

# Start webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

# ROI (centered square)
roi_size = 200
x1 = (frame_width // 2) - (roi_size // 2)
y1 = (frame_height // 2) - (roi_size // 2)
x2 = x1 + roi_size
y2 = y1 + roi_size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresh, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(reshaped, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Show prediction
    cv2.putText(frame, f"Digit: {digit} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Live Digit Recognition", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

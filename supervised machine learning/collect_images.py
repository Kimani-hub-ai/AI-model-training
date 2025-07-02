import cv2
import os

digit = input("Enter digit (0–9): ").strip()
save_dir = os.path.join("data", digit)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
target_count = 200  # Images per digit

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    roi = frame[100:300, 100:300]

    # Show count
    cv2.putText(frame, f"Digit: {digit}  Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capture Digit", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (28, 28))
        filepath = os.path.join(save_dir, f"{count}.png")
        cv2.imwrite(filepath, roi_resized)
        count += 1

    if key == ord('q') or count >= target_count:
        break

cap.release()
cv2.destroyAllWindows()

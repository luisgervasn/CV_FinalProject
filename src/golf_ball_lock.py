import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

correct_sequence = ["Golf Ball", "Golf Ball", "Golf Ball", "Golf Ball"]
detected_sequence = []
messages = ["❌ Secuencia incorrecta. Bloqueo activado."]

def decode_pattern(label):
    global detected_sequence
    detected_sequence.append(label)
    if len(detected_sequence) > 4:
        detected_sequence.pop(0)
    if len(detected_sequence) == 4 and detected_sequence == correct_sequence and messages[-1] == "❌ Secuencia incorrecta. Bloqueo activado.":
        print("✅ Secuencia correcta. Desbloqueo permitido.")
        messages.append("✅ Secuencia correcta. Desbloqueo permitido.")
        

        detected_sequence.clear()

def detect_golf_ball(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=30
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(frame, "Golf Ball", (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(circles) == 4:
            decode_pattern("Golf Ball")
        else:
            detected_sequence.clear()
            if messages[-1] != "❌ Secuencia incorrecta. Bloqueo activado.":
                print("❌ Secuencia incorrecta. Bloqueo activado.")
                messages.append("❌ Secuencia incorrecta. Bloqueo activado.")

    return frame

if __name__ == "__main__":
    while True:
        frame = picam2.capture_array()
        result = detect_golf_ball(frame)
        cv2.imshow("Golf Ball Detector", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

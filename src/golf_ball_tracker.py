import cv2
import numpy as np
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("tracking_output.avi", fourcc, 10.0, (640, 480))

fps = 10 
max_frames = 100  # Queremos 10 segundos
frame_count = 0
trajectory = [] 
tracker_init = False
last_position = None
tracker = None

detected_sequence = []

def detect_balls(frame):
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
        maxRadius=50
    )
    detected = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detected.append((x, y, r))
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.putText(frame, "Golf Ball", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return detected


def find_pink_ball(frame, balls):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([145, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    for (x, y, r) in balls:
        avg_color = cv2.mean(mask[y - r:y + r, x - r:x + r])[0]
        if avg_color > 0:
            return (x, y, r)
    return None


def is_pink(frame, bbox):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([145, 50, 50])
    upper_pink = np.array([170, 255, 255])
    x, y, w, h = [int(v) for v in bbox]
    roi = hsv[y:y + h, x:x + w]
    mask = cv2.inRange(roi, lower_pink, upper_pink)
    avg_color = cv2.mean(mask)[0]
    return avg_color > 0


def decode_pattern(frame, balls):
    global tracker, tracker_init, last_position
    if len(balls) == 4:
        pink_ball = find_pink_ball(frame, balls)
        if pink_ball is not None:
            bbox = (pink_ball[0] - pink_ball[2], pink_ball[1] - pink_ball[2], pink_ball[2] * 2, pink_ball[2] * 2)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            tracker_init = True
            last_position = (pink_ball[0], pink_ball[1])
            print("Tracker iniciado sobre la bola rosa.")
    else:
        tracker_init = False


if __name__ == "__main__":
    start_time = time.time()
    while frame_count < max_frames:
        frame = picam2.capture_array()
        detected_balls = detect_balls(frame)
        if not tracker_init:                        # El tracker se activa si hay 4 bolas visibles
            decode_pattern(frame, detected_balls)
        if tracker_init:                            # Sigue la bola rosa si y solo si el tracker esta iniciado
            success, box = tracker.update(frame)
            if success and is_pink(frame, box):          # Solo se trackea la bola rosa
                (x, y, w, h) = [int(v) for v in box]
                center = (x + w // 2, y + h // 2)
                trajectory.append(center)
                last_position = center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                print("El tracker perdió la bola rosa. Buscando...")
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is not None and trajectory[i] is not None:
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)   # Pinta la trayectoria de la bola
        video_writer.write(frame)
        frame_count += 1
        cv2.imshow("Golf Ball Tracker", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video guardado: {frame_count} frames a {fps} FPS. Duración real: {frame_count / fps:.2f} segundos.")

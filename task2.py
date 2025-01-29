import cv2
import numpy as np
import mediapipe as mp
import random

# Константы
N = 50  # Число шариков
W, H = 640, 480  # Размеры экрана
COLORS = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(N)
]
SIZES = [random.randint(10, 20) for _ in range(N)]
POSITIONS = [
    np.array([random.randint(size, W - size), random.randint(size, H - size)])
    for size in SIZES
]
SPEEDS = [np.array([0, 0], dtype=float) for _ in range(N)]

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Камера
cap = cv2.VideoCapture(0)

# Главный цикл
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадров
    frame = cv2.flip(frame, 1)  # Зеркально перевернуть по горизонтали

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    prev_frame_gray = frame_gray

    # Обнаружение лица и рук
    results_hands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results_face = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Преобразование потока вектора
    flow_magnitude = np.linalg.norm(flow, axis=2)
    flow_angle = np.arctan2(flow[..., 1], flow[..., 0])

    # Движение шариков
    for i in range(N):
        x, y = POSITIONS[i]
        r = SIZES[i]

        # Учет оптического потока
        mask = np.zeros_like(flow_magnitude, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        mask_flow = flow[mask == 255]
        avg_flow = (
            np.mean(mask_flow, axis=0) if len(mask_flow) > 0 else np.array([0, 0])
        )

        # Физика движения
        force = avg_flow
        mass = 0.5
        SPEEDS[i] += force / mass
        SPEEDS[i] *= 0.95  # Затухание
        POSITIONS[i] += SPEEDS[i].astype(int)

        # Контроль границ
        POSITIONS[i][0] = max(r, min(W - r, POSITIONS[i][0]))
        POSITIONS[i][1] = max(r, min(H - r, POSITIONS[i][1]))

    # Отображение
    for i in range(N):
        x, y = POSITIONS[i]
        r = SIZES[i]
        cv2.circle(frame, (x, y), r, COLORS[i], -1)

    # Показ изображения
    cv2.imshow("Optical Flow Bubbles", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Завершение
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np


# Функция для поиска цветного объекта в BGR пространстве
def find_colored_object_bgr(frame, lower_color, upper_color):
    # Создание маски для поиска объектов в заданном диапазоне цветов BGR
    mask = cv2.inRange(frame, lower_color, upper_color)

    # Поиск контуров на бинарной маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


# Основной цикл для захвата видео с камеры
def capture_from_camera_bgr():
    cap = cv2.VideoCapture(0)  # Открытие камеры (номер 0)

    if not cap.isOpened():
        print("Не удается открыть камеру")
        return

    # Размеры кадров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Размер кадра: {width} x {height}")

    cv2.namedWindow("Video")
    cv2.namedWindow("Mask")

    # Диапазон BGR для красного объекта
    lower_color = np.array([0, 0, 150])  # Нижняя граница для красного
    upper_color = np.array([100, 100, 255])  # Верхняя граница для красного

    while True:
        ret, frame = cap.read()  # Чтение нового кадра

        if not ret:
            print("Не удается прочитать кадр")
            break

        # Поиск красных объектов
        contours, mask = find_colored_object_bgr(frame, lower_color, upper_color)

        # Отрисовка прямоугольников вокруг найденных объектов
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Игнорируем мелкие объекты
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображение оригинального видео и маски
        cv2.imshow("Video", frame)
        cv2.imshow("Mask", mask)

        # Ожидание нажатия клавиши 'Esc' для выхода
        if cv2.waitKey(30) == 27:
            print("Нажата клавиша 'Esc'. Выход.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_from_camera_bgr()

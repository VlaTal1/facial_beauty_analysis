import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

while True:
    # Считываем кадр с камеры
    ret, img = cap.read()
    frame = cv2.flip(img, 1)
    height, width, _ = frame.shape

    if not ret:
        break

    # Преобразуем кадр в RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаруживаем лица на кадре
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks is not None:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                if i in [23, 253, 243, 463, 48, 278, 61, 291, 0, 152, 17, 12, 2, 10, 107, 336, 227, 447, 46, 55, 285,
                         276, 61, 17, 291, 105, 334]:
                    cv2.circle(frame, (x, y), 1, (100, 100, 0), -1)
                    cv2.putText(frame, str(i), (x, y), 0, 0.3, (0, 0, 0))

    # Отображаем результат
    cv2.imshow("Face Landmarks Detection", frame)

    # Для выхода из цикла нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()

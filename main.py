from ultralytics import YOLO  # Импортируем класс YOLO из библиотеки ultralytics для обнаружения объектов
import cv2  # Импортируем библиотеку OpenCV для работы с изображениями
import cvzone  # Импортируем библиотеку cvzone для дополнительных функций работы с изображениями

# Инициализируем захват видеопотока с камеры
cap = cv2.VideoCapture("/home/denis/PycharmProjects/MindSetTestWork/data/video/443_motorway_with_cars_1.mp4")
# Загружаем модель YOLO
models = YOLO("../YoloModels/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic Light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "Snowboard", "Sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]


# Начинаем бесконечный цикл для обработки видеопотока
while True:
    success, img = cap.read()  # Считываем кадр из видеопотока success булево значение img картинка массив
    result = models(img, stream=True)  # Применяем модель YOLO к кадру для обнаружения объектов

    # Обрабатываем результаты обнаружения
    for r in result:
        boxes = r.boxes  # Получаем координаты областей, где найдены объекты
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # Извлекаем координаты прямоугольника
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Преобразуем координаты в целые числа

            # Вычисляем ширину и высоту прямоугольника
            w, h = x2 - x1, y2 - y1


            conf = box.conf[0]
            cls = int(box.cls[0])
            correntClass = classNames[cls]

            if correntClass == "car" or correntClass == "truck" or correntClass == "bus" and conf >0.3:

                cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3 )
                cvzone.cornerRect(img, (x1, y1, w, h), l=6, t=1)

    # Отображаем изображение с нарисованными прямоугольниками
    cv2.imshow('image', img)
    cv2.waitKey(1)  # Ожидаем нажатие клавиши для продолжения обработки

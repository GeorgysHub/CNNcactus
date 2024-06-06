import cv2
import numpy as np
def predict_image(model, img_path, categories, img_size=224, confidence_threshold=0.5):
    # Чтение и предобработка изображения
    img = cv2.imread(img_path)
    if img is None:
        return "Ошибка: изображение не найдено."

    img_resized = cv2.resize(img, (img_size, img_size))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Получение предсказания
    prediction = model.predict(img_array)
    max_confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)

    # Проверка уверенности предсказания
    if max_confidence < confidence_threshold:
        return "Вы выбрали неподходяющую картинку"
    else:
        return categories[predicted_class[0]], max_confidence

def draw_prediction(img_path, prediction, confidence=None):
    img = cv2.imread(img_path)
    if img is not None:
        text = prediction
        if confidence is not None:
            text += f" ({confidence*100:.2f}%)"

        # Наложение текста на изображение
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
    else:
        return "Ошибка: изображение не найдено."



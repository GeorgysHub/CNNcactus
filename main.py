import os
import tensorflow as tf
from preparation_data import load_data
from model import build_model
from preparation import fit_model
from check import predict_image, draw_prediction

def main():
    action = input("Введите 'train' для тренировки модели или 'load' для загрузки модели: ").strip().lower()

    if action == 'train':
        data_path = input("Введите путь к вашему датасету: ")
        if not os.path.exists(data_path):
            print(f"Ошибка: Путь {data_path} не существует.")
            return

        categories = ["Astrophytum asteria", "dragon", "Ferocactus", "Gigantea"]
        print(f"Используемый путь к данным: {data_path}")
        print(f"Категории: {categories}")

        data, labels = load_data(data_path, categories)
        model = build_model(num_classes=len(categories))

        model = fit_model(model, data, labels, batch_size=32, epochs=100)
        model.save('cactus_model1.keras')
        print("Модель обучена и сохранена как 'cactus_model.keras'")

    elif action == 'load':
        # Load the model
        model = tf.keras.models.load_model('cactus_model.keras')
        print("Модель загружена из 'cactus1_model.keras'")

    else:
        print("Неверное действие. Пожалуйста, введите 'train' или 'load'.")
        return

    # Testing loop
    categories = ["Astrophytum asteria", "dragon", "Ferocactus", "Gigantea"]
    while True:
        img_path = input("Введите путь к тестовому изображению: ")
        if img_path.lower() == 'exit':
            print("Выход...")
            break
        if not img_path:
            print("Ошибка: Пожалуйста, введите действительный путь к тестовому изображению.")
            continue
        if not os.path.exists(img_path):
            print(f"Ошибка: Путь к изображению {img_path} не существует.")
            continue
        result = predict_image(model, img_path, categories)
        print(f"Распознанный объект: {result}")
        draw_prediction(img_path, result)

if __name__ == "__main__":
    main()



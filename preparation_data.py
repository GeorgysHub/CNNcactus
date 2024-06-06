import os
import cv2
import numpy as np


def load_data(data_directory, class_names, image_size=224):
    dataset = []
    label_list = []

    for class_name in class_names:
        folder_path = os.path.join(data_directory, class_name)
        class_label = class_names.index(class_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Предупреждение: Не удалось прочитать изображение {file_path}.")
                    continue
                resized_image = cv2.resize(image, (image_size, image_size))
                dataset.append(resized_image)
                label_list.append(class_label)
            except Exception as error:
                print(f"Ошибка при обработке изображения {filename}: {error}")

    dataset = np.array(dataset)
    label_list = np.array(label_list)

    return dataset, label_list


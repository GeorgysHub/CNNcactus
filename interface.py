import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tensorflow as tf
from PIL import Image, ImageTk
from preparation_data import load_data
from model import build_model
from preparation import fit_model
from check import predict_image

class CactusClassifier:
    def __init__(self, master):
        self.master = master
        master.title("Классификация кактусов")

        # Define style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton",
                        font=('Helvetica', 12),
                        padding=10,
                        relief="flat",
                        background="#2E8B57",
                        foreground="white")
        style.map("TButton",
                  foreground=[('active', 'white'), ('pressed', 'white')],
                  background=[('active', '#3CB371'), ('pressed', '#3CB371')])
        style.configure("TLabel", font=('Helvetica', 16, 'bold'), foreground='#2E8B57')
        style.configure("TFrame", padding=10)

        main_frame = ttk.Frame(master, style="TFrame")
        main_frame.grid(row=0, column=0, padx=20, pady=20)

        self.label = ttk.Label(main_frame, text="Классификация кактусов", style="TLabel")
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.train_button = ttk.Button(main_frame, text="Тренировка модели", command=self.train_model_callback)
        self.train_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        self.load_button = ttk.Button(main_frame, text="Загрузка модели", command=self.load_model_callback)
        self.load_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        self.test_button = ttk.Button(main_frame, text="Определить кактус", command=self.predict_callback)
        self.test_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        self.exit_button = ttk.Button(main_frame, text="Выход", command=master.quit)
        self.exit_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        self.canvas = tk.Canvas(main_frame, width=400, height=300, bg='white', bd=2, relief='solid')
        self.canvas.grid(row=5, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(main_frame, text="", font=('Helvetica', 14), foreground='darkred')
        self.result_label.grid(row=6, column=0, columnspan=2, pady=10)

        self.info_label = ttk.Label(main_frame, text="", font=('Helvetica', 12), foreground='blue')
        self.info_label.grid(row=7, column=0, columnspan=2, pady=10)

        self.model = None
        self.categories = ["Astrophytum asteria", "dragon", "Ferocactus", "Gigantea"]

    def train_model_callback(self):
        data_path = filedialog.askdirectory(title="Выберите папку")
        if not data_path:
            messagebox.showerror("Ошибка", "Папка не выбрана.")
            return

        try:
            data, labels = load_data(data_path, self.categories)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неудачная загрузка данных: {e}")
            return

        self.model = build_model(num_classes=len(self.categories))
        self.model = fit_model(self.model, data, labels, batch_size=64, epochs=100)
        self.model.save('cactus_model.keras')
        self.info_label.config(text="Модель прошла тренировку и записана как 'cactus_model.keras'")

    def load_model_callback(self):
        model_path = filedialog.askopenfilename(title="Выберите модель", filetypes=[("Keras Models", "*.keras")])
        if not model_path:
            messagebox.showerror("Ошибка", "Модель не была выбрана.")
            return

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.info_label.config(text="Модель загружена 'cactus_model.keras'")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки модели: {e}")

    def predict_callback(self):
        if not self.model:
            messagebox.showerror("Ошибка", "Модель не загружена.")
            return

        img_path = filedialog.askopenfilename(title="Выберите изображения для определения", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not img_path:
            messagebox.showerror("Ошибка", "Изображение не выбрано.")
            return

        try:
            result = predict_image(self.model, img_path, self.categories)
            self.result_label.config(text=f"Распознан объект: {result}")
            self.display(img_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознания: {e}")

    def display(self, img_path):
        img = Image.open(img_path)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

def main():
    root = tk.Tk()
    CactusClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()

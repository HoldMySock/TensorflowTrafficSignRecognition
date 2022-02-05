import os
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pickle

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

import numpy as np

with open('test.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)
y_test = np.argmax(y_test, axis=1)

model = load_model("model2.h5")
prediction = np.argmax(model.predict(x_test), axis=1)
accuracy = float(accuracy_score(y_test, prediction.round()))
print('Точність моделі при тестувані становить {:.2f}'.format(accuracy * 100), "%")

# словник назв
classes = {1: 'Обмеження максимальної швидкості(20км/год)',
           2: 'Обмеження максимальної швидкості (30км/год)',
           3: 'Обмеження максимальної швидкості (50км/год)',
           4: 'Обмеження максимальної швидкості (60км/год)',
           5: 'Обмеження максимальної швидкості (70км/год)',
           6: 'Обмеження максимальної швидкості (80км/год)',
           7: 'Кінець обмеження максимальної швидкості (80км/год)',
           8: 'Обмеження максимальної швидкості (100км/год)',
           9: 'Обмеження максимальної швидкості (120км/год)',
           10: 'Обгін заборонено',
           11: 'Обгін вантажним автомобілям заборонено',
           12: 'Перехрещення з другорядною дорогою',
           13: 'Головна дорога',
           14: 'Дати дорогу',
           15: 'Проїзд без зупинки заборонено',
           16: 'Рух заборонено',
           17: 'Рух вантажних автомобілів заборонено',
           18: 'В''їзд заборонено',
           19: 'Інша небезпека (аварійно-небезпечна ділянка)',
           20: 'Небезпечний поворот ліворуч',
           21: 'Небезпечний поворот праворуч',
           22: 'Декілька поворотів',
           23: 'Нерівна дорога',
           24: 'Слизька дорога',
           25: 'Звуження дороги',
           26: 'Дорожні роботи',
           27: 'Світлофорне регулювання',
           28: 'Пішоходний перехід',
           29: 'Діти',
           30: 'Виїзд велосипедистів',
           31: 'Сніг',
           32: 'Дикі тварини',
           33: 'Кінець усіх заборон і обмежень',
           34: 'Рух праворуч',
           35: 'Рух ліворуч',
           36: 'Рух прямо',
           37: 'Рух прямо або праворуч',
           38: 'Рух прямо або ліворуч',
           39: 'Рух праворуч',
           40: 'Об’їзд перешкоди з лівого боку',
           41: 'Круговий рух',
           42: 'Кінець заборони обгону',
           43: 'Кінець заборони обгону вантажним автомобілям'}

# GUI
top = tk.Tk()
top.geometry('1600x900')
top.title('Розпізнавання дорожніх знаків')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 18, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground='#364196', text=sign)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path).convert('RGB')
        uploaded.save(r'D:\pycharmproj\TensorflowTrafficSignRecognition\temp\temp_converted_image.png')
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        classify(r'D:\pycharmproj\TensorflowTrafficSignRecognition\temp\temp_converted_image.png')
    except:
        pass


def initialize_gui():
    learn_b = Button(top, text="Оновити модель", command=lambda: os.system('python LearnModel.py'), padx=10, pady=5)
    learn_b.configure(background='#364196', foreground='white', font=('arial', 13, 'bold'))
    learn_b.place(relx=0.05, rely=0.85)

    exit_button = Button(top, text="Вихід", command=top.destroy, font=('arial', 14, 'bold'), padx=10, pady=5)
    exit_button.configure(background='#364196', foreground='white')
    exit_button.place(relx=0.85, rely=0.85)

    upload = Button(top, text="Завантажити знак", command=upload_image, pady=5)
    upload.configure(background='#364196', foreground='white', font=('arial', 10, 'bold'))

    upload.pack(side=BOTTOM, pady=70)
    sign_image.pack(side=BOTTOM, expand=True)
    label.pack(side=BOTTOM, expand=True)
    heading = Label(top, text="Розпізнавання дорожніх знаків", pady=10, font=('Times New Roman', 22, 'bold'))
    heading.configure(background='#CDCDCD', foreground='#364196')
    heading.pack()
    top.mainloop()


initialize_gui()

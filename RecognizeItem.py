import time
import cv2
import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow.keras as tfk
from tensorflow.python.keras.callbacks import TensorBoard

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
batch_size = 16
pool_size = (2, 2)
inputShape = (IMG_WIDTH, IMG_HEIGHT, 3)
Name = "trafficsSignsModel-{}".format(int(time.time()))


def main():
	# Перевірка аргументів командної строки
	if len(sys.argv) not in [1, 3]:
		sys.exit("Usage: python RecognizeItem.py data_directory [model.h5]")

	images, labels = load_data(os.path.dirname(sys.argv[0]))

	labels = tfk.utils.to_categorical(labels)
	x_train, x_test, y_train, y_test = train_test_split(
		np.array(images), np.array(labels), test_size=TEST_SIZE)

	model = get_model()
	model.summary()
	tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))
	history = model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[tensorboard])

	# Оцінка роботи нейронної мережі
	_, acc_train = model.evaluate(x_train, y_train, verbose=2)
	_, acc_test = model.evaluate(x_test, y_test, verbose=2)
	print('Точність навчання : %.3f, Точність тестування: %.3f' % (acc_train, acc_test))

	# Зберігання моделі
	if len(sys.argv) == 1:
		folder_name = os.path.dirname(sys.argv[0])
		print(os.path.join(folder_name, "model1.h5"))
		model.save(os.path.join(folder_name, "model1.h5"))
		print(f"Model saved to {folder_name}.")
		plt.figure(0)
		plt.plot(history.history['loss'], label='training loss')
		plt.title("Втрати")
		plt.xlabel("епохи")
		plt.ylabel("втрати")
		plt.legend()
		plt.show()

		plt.figure(1)
		plt.plot(history.history['accuracy'], label='training accuracy ')
		plt.title("Точність")
		plt.xlabel("епохи")
		plt.ylabel("точність")
		plt.legend()
		plt.show()


def load_data(data_path):
	data = []
	labels = []
	for i in range(NUM_CATEGORIES):
		path = os.path.join(r"D:\pycharmproj\TensorflowTrafficSignRecognition\gtsrb\Train", str(i))
		images = os.listdir(path)
		for j in images:
			try:
				image = cv2.imread(os.path.join(path, j))
				image_from_array = Image.fromarray(image, 'RGB')
				resized_image = image_from_array.resize((IMG_HEIGHT, IMG_WIDTH))
				data.append(np.array(resized_image))
				labels.append(i)
			except AttributeError:
				print("Помилка завантаження зображення!")

	images_data = (data, labels)
	return images_data


def get_model():
	# ініціалізація моделі
	model = Sequential()
	ch_dimension = -1
	model.add(Conv2D(8, (5, 5), padding="same", input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=ch_dimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(NUM_CATEGORIES))
	model.add(Activation("softmax"))

	# компілювання моделі
	model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

	return model


if __name__ == "__main__":
	main()

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Слой нейронов
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # инициальизация случайных весов от 0 до 1
		self.biases = np.zeros(n_neurons)  # инициализация биасов равных 0
	
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases  # обработка согласно весам и биасам


# линейная активация
class Activation_ReLu:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)  # если меньше нуля, вернуть 0


# софтмакс активация
class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # вычисление экспоненты батча инпутов, где из каждого элемента батча вычитается максимальный элемент строки
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # деление каждого элемента получившегося батча на сумму всех элементов строки
		self.output = probabilities


# вычисление лосса
class Loss:
	# результат softmax, массив целевых данных
	def calculate(self, output, y):
		sample_loss = self.forward(output, y)  # прогнать данные по функции forward соответсвующего типа лосса
		data_loss = np.mean(sample_loss)  # вычислить средний loss всего батча
		return data_loss


class Loss_categoricalCrossEntropy(Loss):
	def forward(self, y_predictions, y_true):
		samples = len(y_predictions)
		y_predictions = np.clip(y_predictions, 1e-7, 1 - 1e-7)  # установить данные в промежуток между 1е-7 до 1-1е-7 (чтобы избежать логарфима 0)
		
		if len(y_true.shape) == 1:
			correct_confidences = y_predictions[range(samples), y_true]  # взять данные из батча y_predictions под номерами из массива y_true
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_predictions * y_true, axis=1)  # перемножить батчи и проссумировать строки получившихся (https://youtu.be/levekYbxauw?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&t=777)
		
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods


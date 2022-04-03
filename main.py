import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Слой нейронов
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Инициализация случайных весов от 0 до 1.
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Инициализация биасов равных 0.
        self.biases = np.zeros(n_neurons)
    
    def forward(self, inputs):
        # Умножение всех инпутов на их веса плюс биас.
        self.output = np.dot(inputs, self.weights) + self.biases


# Линейная активация

# x < 0   =>   y = 0
# x > 0   =>   y = x
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Софтмакс активация

# 1 Из всех элементов батча вычитается максимальный => все эл. < 0, макс. = 0
# 2 Вычислить экспоненту всех элементов.
# 3 Найти долю каждого элементов в строке батча,
#   поделив каждый элемент на сумму остальных.
class Activation_Softmax:
    def forward(self, inputs):
        # 1, 2
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # 3
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Вычисление лосса.
# На вход результат софтмакс и батч строк по типу [0, 0, 1, 0],
# где 1 - желаемый выход нейронной сети (максимальное значение софтмакс)
# На выход - цифра, обозначающая отклонение от правды (чем больше, тем хуже)
class Loss:
    def calculate(self, y_predictions, y_true):
        # Прогнать данные по функции forward соответсвующего типа лосса.
        sample_loss = self.forward(y_predictions, y_true)
        # Вычислить средний лосс всего получившегося батча.
        data_loss = np.mean(sample_loss)
        return data_loss


class Loss_categoricalCrossEntropy(Loss):
    def forward(self, y_predictions, y_true):
        # Установить данные в промежуток между 1е-7 до 1-1е-7
        # чтобы избежать логарфима 0.
        y_predictions = np.clip(y_predictions, 1e-7, 1 - 1e-7)
        
        # Если искомые данные выглядят так [2, 1]
        if len(y_true.shape) == 1:
            # Создать строку с данными из массива y_predictions
            # под номерами y_true.
            correct_confidences = y_predictions[range(len(y_true)), y_true]
        
        # Если искомые данные выглядят так [[0, 1, 0],[1, 0, 0]]
        elif len(y_true.shape) == 2:
            # Перемножить батчи (ненужные умножатся на 0) и вернуть массив
            # с суммами элементов всех строк.
            correct_confidences = np.sum(y_predictions * y_true, axis=1)
        
        # Взять отрицательный логарфим всех элементов из получившегося
        # массива нужных нам данных.
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Вычисление точности нейронной сети путем
# Из батча данных получившихся на выходе ищем процент совпавших с целевыми.
def accuracy(inputs, targets):
    # Вернуть массив с номерами максимальных чисел каждой строки батча
    predictions = np.argmax(inputs, axis=1)
    # Найти процент сходящихся элементов этого массива с массивом targets
    accuracy = np.mean(predictions == targets)
    return int(accuracy * 100)

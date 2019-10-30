import math
import random

import numpy as np

BIAS_NODE = 1
X_TRAIN_FILE_NAME = 'train/X_train.csv'
X_TEST_FILE_NAME = 'test/X_test.csv'
Y_TRAIN_FILE_NAME = 'train/Y_train.csv'


class RedNeuronal:
    def __init__(self, n_entradas, n_ocultos, n_salidas):
        self.n_entradas = n_entradas + BIAS_NODE
        self.n_ocultos = n_ocultos
        self.n_salidas = n_salidas
        self.y_test = []
        self.generar_pesos_entrada()

        self.ultimo_cambio_entrada = np.zeros((self.n_entradas, self.n_ocultos))
        self.ultimo_cambio_salida = np.zeros((self.n_ocultos, self.n_salidas))

        self.generar_activaciones()

    def generar_pesos_entrada(self):
        self.pesos_entrada = np.zeros((self.n_entradas, self.n_ocultos))
        self.pesos_salida = np.zeros((self.n_ocultos, self.n_salidas))

        for i in range(self.n_entradas):
            for j in range(self.n_ocultos):
                self.pesos_entrada[i][j] = randomEntre(-0.5, 0.5)
        for j in range(self.n_ocultos):
            for k in range(self.n_salidas):
                self.pesos_salida[j][k] = randomEntre(-2.0, 2.0)

    def generar_activaciones(self):
        self.activacion_de_entrada = [1.0] * self.n_entradas
        self.activacion_de_ocultos = [1.0] * self.n_ocultos
        self.activacion_de_salidas = [1.0] * self.n_salidas

    def actualizar(self, entradas):
        self.activaciones_de_entradas(entradas)
        self.activaciones_de_ocultos()
        self.activaciones_de_salidas()

        return self.activacion_de_salidas[:]

    def activaciones_de_salidas(self):
        for k in range(self.n_salidas):
            suma = 0.0
            for j in range(self.n_ocultos):
                suma = suma + self.activacion_de_ocultos[j] * self.pesos_salida[j][k]
            self.activacion_de_salidas[k] = sigmoid(suma)

    def activaciones_de_ocultos(self):
        for j in range(self.n_ocultos):
            suma = 0.0
            for i in range(self.n_entradas):
                suma += self.activacion_de_entrada[i] * self.pesos_entrada[i][j]
            self.activacion_de_ocultos[j] = sigmoid(suma)

    def activaciones_de_entradas(self, inputs):
        for i in range(self.n_entradas - BIAS_NODE):
            self.activacion_de_entrada[i] = inputs[i]

    def backpropagation(self, objetivos, learning_rate, factor_momento):
        if len(objetivos) != self.n_salidas:
            raise ValueError('wrong number of target values')

            # calculate error terms for output
        salidas_delta = [0.0] * self.n_salidas
        for k in range(self.n_salidas):
            error = objetivos[k] - self.activacion_de_salidas[k]
            salidas_delta[k] = dsigmoid(self.activacion_de_salidas[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.n_ocultos
        for j in range(self.n_ocultos):
            error = 0.0
            for k in range(self.n_salidas):
                error += salidas_delta[k] * self.pesos_salida[j][k]
            hidden_deltas[j] = dsigmoid(self.activacion_de_ocultos[j]) * error

        for j in range(self.n_ocultos):
            for k in range(self.n_salidas):
                change = salidas_delta[k] * self.activacion_de_ocultos[j]
                self.pesos_salida[j][k] = self.pesos_salida[j][k] + learning_rate * change + factor_momento * self.ultimo_cambio_salida[j][k]
                self.ultimo_cambio_salida[j][k] = change

        for i in range(self.n_entradas):
            for j in range(self.n_ocultos):
                change = hidden_deltas[j] * self.activacion_de_entrada[i]
                self.pesos_entrada[i][j] = self.pesos_entrada[i][j] + learning_rate * change + factor_momento * self.ultimo_cambio_salida[i][j]
                self.ultimo_cambio_salida[i][j] = change

        error = 0.0
        # 0.5 es el Learning rate
        for k in range(len(objetivos)):
            error += 0.5 * (objetivos[k] - self.activacion_de_ocultos[k]) ** 2
        return error

    def test(self, entrenamiento):
        for row in entrenamiento:
            if self.actualizar(row[0])[0] > 0.5:
                prediccion = 1
            else:
                prediccion = 0

            self.y_test.append(prediccion)

    def entrenamiento(self, entrenamiento, iteraciones=1000, lerning_rate=0.1, factor_momento=0.01):
        for i in range(iteraciones):
            error = 0.0
            for row in entrenamiento:
                entradas = row[0]
                objetivos = row[1]
                self.actualizar(entradas)
                error += self.backpropagation(objetivos, lerning_rate, factor_momento)
            print('>iteracion de entrenamiento=%d, learning_rate=%.3f,factor_momento=%.3f error=%.3f'
                  % (i, lerning_rate, factor_momento, error))

def ejecutar():
    red_neuronal = RedNeuronal(5, 8, 1)
    random.seed(0)

    x_train = obtener_datos_de_archivo(X_TRAIN_FILE_NAME)
    x_test = obtener_datos_de_archivo(X_TRAIN_FILE_NAME)
    y_train = obtener_datos_de_archivo(X_TRAIN_FILE_NAME)

    entrenamiento = []
    for i in range(x_train.shape[0]):
        entrenamiento.append([x_train[i], [y_train[i]]])

    red_neuronal.entrenamiento(entrenamiento, 100)
    # test it
    test = []

    for i in range(x_test.shape[0]):
        test.append([x_test[i]])
    red_neuronal.test(test)
    np.savetxt('Y_test.csv', [red_neuronal.y_test], delimiter='\n', fmt='%d')


def obtener_datos_de_archivo(nombre_archivo):
    dataX = np.genfromtxt(nombre_archivo, delimiter=",")
    x_train = np.array(dataX)
    return x_train


def randomEntre(a, b):
    return (b - a) * random.random() + a


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y ** 2


if __name__ == '__main__':
    ejecutar()

# Clasficacion Perceptron
# Angel Isaac Gomez Canales
# 14/02/2024

# El codigo se puede optimizar mucho pero para fines educativos es suficiente

import numpy as np
import matplotlib.pyplot as plt

# Construir la clase Neurona --------------------------------------------------
class Neurona:
    def __init__(self, n_inputs, n_outputs, lr=0.1):
        # Inicializar pesos y bias aleatoriamente entre -1 y 1
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.lr = lr

    def plot_hiperplano(self, xrecta, n_neurona):
        # xrecta es una lista de dos elementos con los puntos inicial y final de la recta
        w1, w2, b = self.w[n_neurona, 0], self.w[n_neurona, 1], self.b[n_neurona]
        plot_l = plt.plot(xrecta, [(-1/w2) * (w1 * xrecta[0] + b), (-1/w2) * (w1 * xrecta[1] + b)], 'k')
        return plot_l
    
    def predict(self, X):  # Pred para un punto
        z = self.w @ X + self.b.ravel()
        return np.sign(z)

    def entrenar(self, X, Y):
        # Graficar estado inicial de la red
        #plot_l, = self.plot_hiperplano([-6, 6])
        #plt.pause(0.25)
        #plt.axis([-6, 6, -6, 6])
        i = 0
        bandera1 = True
        bandera2 = True
        exitos1 = 0
        exitos2 = 0
        plot_l1, = self.plot_hiperplano([-6, 8], 0)
        plot_l2, = self.plot_hiperplano([-6, 8], 1)
        plt.pause(0.25)
        plt.axis([0, 7, 0, 7])
        while bandera1 or bandera2:
            # calcular salida de la red
            y_pred = self.predict(X[:, i])

            # calcular el error
            e1 = Y[0, i] - y_pred[0]
            e2 = Y[1, i] - y_pred[1]
            # Si hay error se actualizan los pesos
            if e1 != 0:
                exitos1 = 0
                self.w[0, :] += (self.lr * e1) * X[:, i]
                self.b[0] += self.lr * e1
                plot_l1.remove()
                plot_l1, = self.plot_hiperplano([-6, 8], 0)
                plt.pause(0.1)
            else:
                exitos1 += 1
                # El entrenamiento acaba cuando todos son clasificados de manera correcta
                if exitos1 == Y.shape[1]:
                    bandera1 = False
            if e2 != 0:
                exitos2 = 0
                self.w[1, :] += (self.lr * e2) * X[:, i]
                self.b[1] += self.lr * e2
                plot_l2.remove()
                plot_l2, = self.plot_hiperplano([-6, 8], 1)
                plt.pause(0.1)
            else:
                exitos2 += 1
                # El entrenamiento acaba cuando todos son clasificados de manera correcta
                #self.plot_hiperplano([-6, 6])
                if exitos2 == Y.shape[1]:
                    bandera2 = False
            i += 1
            if i > X.shape[1]-1:
                i = 0

# Generar los puntos a clasificar -------------------------------------------
n = 20  # Puntos a generar por clase
centros = np.array([[3, 5],  
                    [5, 3],
                    [3, 1],
                    [1, 3]])  # Centros de cada clase alrededor de los que se generaran los puntos

X = np.zeros((2, len(centros) * n))
Y = np.zeros((2, len(centros) * n))

# Asignar la etiqueta a cada clase
Y[:, 0: n*1] = np.array([[1, 1]] * n).T
Y[:, n*1: n*2] = np.array([[1, -1]] * n).T
Y[:, n*2: n*3] = np.array([[-1, -1]] * n).T
Y[:, n*3: n*4] = np.array([[-1, 1]] * n).T

# Convertir las etiquetas a enteros (solo para visualizacion)
def label2int(Y):
    mapeo = {(1, 1): 0, (1, -1): 1, (-1, -1): 2, (-1, 1): 3}
    Y = [mapeo[tuple(y)] for y in Y.T]
    return Y

Y_int = label2int(Y)

# Generar los n puntos para cada clase
for i, centro in enumerate(centros):
    X[:, i * n: (i+1) * n] = centro.reshape((-1, 1)) + 1.2 * np.random.rand(2, n)

# Visualizar el grafico original (deseado)
plt.figure()
sc = plt.scatter(X[0, :], X[1, :])
plt.title("Datos originales")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

# En este caso no es necesario normalizar porque x1 y x2 tienen rangos similares

# Separar datos en entrenamiento y prueba
idx = np.random.permutation(X.shape[1])  # Indices mezclados
n_train = int(0.8 * X.shape[1])  # 80% para entrenar y 20% prueba
X_train = X[:, idx[:n_train]]
Y_train = Y[:, idx[:n_train]]
X_test = X[:, idx[n_train:]]
Y_test = Y[:, idx[n_train:]]

# Crear Neurona dos entradas y dos salidas --------------------------------------
neurona = Neurona(2, 2, lr=1)
neurona.entrenar(X_train, Y_train)

for i, x in enumerate(X_test.T):
    y = neurona.predict(x)
    e = np.sum(Y_test[:, i] - y)
    print(f"Error del dato {i}: {e}")

Ypred = np.array([neurona.predict(x) for x in X.T]).T
plt.figure()
plt.scatter(X[0, :], X[1, :], c=label2int(Ypred), cmap='rainbow')
plt.title("Predicciones de la red")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
neurona.plot_hiperplano([-6, 8], 0)
neurona.plot_hiperplano([-6, 8], 1)
plt.axis([0, 7, 0, 7])
plt.show()
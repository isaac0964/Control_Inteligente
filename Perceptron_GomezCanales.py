# Perceptron
# Angel Isaac Gomez Canales
# 14/02/2024

import numpy as np
import matplotlib.pyplot as plt

# Paso 1. Datos de Entrada. 
X = np.array([[2, 3, 4, 1, 4, 6, 0, 2, 3, 1, 1, -1, -1, -2, -2, -2, -3, -6],
              [2, 6, 2, -2, -4, -6, -6, -6, 4, 6, 5, -1, -3, 1, 2, 3, -6, 6]])

Y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Normalizar los datos entre 0 y 1
X_norm = X#(X - np.min(X, axis=1).reshape(-1, 1)) / ((np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1)) 

# Plotear para ver los datos
plt.figure()
plt.title("Datos originales")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.scatter(X[0, :], X[1, :], c=Y, cmap="rainbow")

      
# Separar datos en entrenamiento y prueba
idx = np.random.permutation(X.shape[1])  # Indices mezclados
n_train = int(0.8 * X.shape[1])  # 80% para entrenar y 20% prueba
X_train = X_norm[:, idx[:n_train]]
Y_train = Y[idx[:n_train]]
X_test = X_norm[:, idx[n_train:]]
Y_test = Y[idx[n_train:]]

# Construir la Neurona
class Neurona:
    def __init__(self, n_inputs, n_outputs, lr=0.1):
        # Inicializar pesos y bias aleatoriamente entre -1 y 1
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.lr = lr

    def plot_hiperplano(self, xrecta):
        # xrecta es una lista de dos elementos con los puntos inicial y final de la recta
        w1, w2, b = self.w[0, 0], self.w[0, 1], self.b[0]
        plot_l = plt.plot(xrecta, [(-1/w2) * (w1 * xrecta[0] + b), (-1/w2) * (w1 * xrecta[1] + b)])
        return plot_l
    
    def predict(self, x):  # Pred para un punto
        z = self.w @ x + self.b
        return np.sign(z)

    def entrenar(self, X, Y):
        bandera = True
        exitos = 0
        # Graficar estado inicial de la red
        plot_l, = self.plot_hiperplano([-6, 6])
        plt.pause(0.25)
        plt.axis([-6, 6, -6, 6])
        
        while bandera:
            for i in range(X.shape[1]):
                # calcular salida de la red
                y_pred = self.predict(X[:, i])

                # calcular el error
                e = Y[i] - y_pred
                # Si hay error se actualizan los pesos
                if e != 0:
                    exitos = 0
                    self.w += self.lr * e * X[:, i]
                    self.b += self.lr * e
                    plot_l.remove()
                    plot_l, = self.plot_hiperplano([-6, 6])
                    plt.pause(0.2)
                else:
                    exitos += 1
                    # El entrenamiento acaba cuandot todos son clasificados de manera correcta
                    if exitos == len(Y):
                        bandera = False

neurona = Neurona(2, 1, lr=0.1)
neurona.entrenar(X_train, Y_train)
plt.show()
# Funcion Logica ~((A v B)^)C Perceptron
# Angel Isaac Gomez Canales
# 14/02/2024

# El codigo se puede optimizar mucho pero para fines educativos es suficiente

import numpy as np
import matplotlib.pyplot as plt

# Construir la Neurona
class Neurona:
    def __init__(self, n_inputs, n_outputs, lr=0.1):
        # Inicializar pesos y bias aleatoriamente entre -1 y 1
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.lr = lr

    def plot_hiperplano(self):
        w1, w2, w3, b = self.w[0, 0], self.w[0, 1], self.w[0, 2], self.b[0]
        # Crear una cuadrícula de puntos en el espacio
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        x, y = np.meshgrid(x, y)
        z = (-w1 * x - w2 * y - b) / w3
        plano = ax.plot_surface(x, y, z, alpha=0.5, color='k')
        return plano
    
    def predict(self, x):  # Pred para un punto
        z = self.w @ x + self.b
        return np.sign(z)

    def entrenar(self, X, Y):
        bandera = True
        exitos = 0
        # Graficar estado inicial de la red
        plano = self.plot_hiperplano()
        ax.axis([0, 1, 0, 1, 0, 1])
        plt.pause(0.25)        
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
                    plano.remove()
                    plano = self.plot_hiperplano()
                    plt.pause(0.2)
                else:
                    exitos += 1
                    # El entrenamiento acaba cuandot todos son clasificados de manera correcta
                    if exitos == len(Y):
                        bandera = False

# Generar datos -------------------------
X = np.array([[1, 1, 1, 1, 0, 0, 0, 0],   # A
              [1, 1, 0, 0, 1, 1, 0, 0],   # B
              [1, 0, 1, 0, 1, 0, 1, 0]])  # C

Y = np.array([-1, 1, -1, 1, -1, 1, 1, 1])

# Graficar los puntos 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0, :], X[1, :], X[2, :], c=Y, cmap='rainbow')
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")

# Crear Neurona tres entradas y dos salidas --------------------------------------
neurona = Neurona(3, 1)
neurona.entrenar(X, Y)
#neurona.plot_hiperplano()

for i in range(X.shape[1]):
    error = Y[i] - neurona.predict(X[:, i])
    print(f"Error del dato {i}: {error}")
plt.show()
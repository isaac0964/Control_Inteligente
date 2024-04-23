# Clase para Red Neuronal Multicapa entrenada con filtro de Kalman
# Angel Isaac Gomez Canales  22/04/2024

import numpy as np
import matplotlib.pyplot as plt

# Definir funciones de activacion
def lineal(z, derivative=False):
    a = z
    if derivative:
        da = np.ones_like(z, dtype=float)
        return a, da
    return a

def logistica(z, alpha=1, derivative=False):
    a = 1 / (1 + np.exp(-alpha * z))
    if derivative:
        da = alpha * a * (1 - a)
        return a, da
    return a

def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 - a) * (1 + a)
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype=float)
        return a, da
    return a

# Crar clase para red multicapa
class MLP_Kalman():
    def __init__(self, n_in, n_h, n_o=1, activacion_oculta=relu, eta=0.1):
        # Asumimos que hay solo una capa oculta, una neurona de salida, y la activacion de la salida lineal
        # Instanciar atributos con listas vacias
        self.n_in = n_in  # Neuronas de entrada
        self.n_h = n_h  #  Neuronas de capa oculta
        self.n_o = n_o  # Neuronas en capa oculta
        self.w = []  # pesos de cada capa
        self.f = activacion_oculta  # funcion de activacion de capa oculta
        self.eta = eta
        
        # Agregar matrices de pesos
        self.w.append(-1 + 2 * np.random.rand(n_h, n_in+1)) 
        self.w.append(-1 + 2 * np.random.rand(n_o, n_h+1))

        # Numero de pesos
        self.nW = (self.n_h  * (self.n_in + 1)) + (self.n_o * (self.n_h +1))

    # Metodo para hacer predicciones
    def predict(self, X):
        a_h = self.f(np.dot(self.w[0][:, :-1], X) + self.w[0][:, -1:])  # Activacion de oculta
        y = np.dot(self.w[1][:, :-1], a_h) + self.w[1][:, -1:]  # Salida de la red
        return y
    
    # Metodo para entrenar la red
    def fit(self, X, Y, epochs=100, p=0.5, q=0, r=0.1):
        print("----------------------- Inicio del Entrenamiento -------------------")
        # Crear Matrices P, Q y R
        P = p * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        Q = q * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        if r == None:
            r = self.eta ** -1
        R = r * np.eye(Y.shape[0])  # Tamaño de Nsalidas x Nsalias (m)
        # Entrenar por n epocs
        for _ in range(epochs):
            # Mezclar los datos al iniciar cada epoca
            idx = np.random.permutation(X.shape[1])
            X = X[:, idx]
            Y = Y[:, idx]

            # Entrenar con cada punto
            for i in range(X.shape[1]):
                # Tomar el punto con el que se entrenara
                X_p = X[:, i].reshape((-1, 1))
                Y_p = Y[:, i]

                # Propagacion hacia adelante
                a_h, da_h = self.f(np.dot(self.w[0][:, :-1], X_p) + self.w[0][:, -1:], derivative=True)  # Activacion de oculta y derivada
                y = np.dot(self.w[1][:, :-1], a_h) + self.w[1][:, -1:]  # Salida de la red

                # Calcular el error
                e = Y_p - y

                # Obtener matriz de derivadas (H)

                # Primero las derivadas de la salida con respecto a los pesos de la capa de salida 
                dw1 = np.vstack((a_h, [1])).ravel()

                # Derivadas de la salida con respecto a los pesos de la capa oculta
                w_phi = self.w[1][:, :-1].T * da_h
                dw0 = np.outer(w_phi, np.vstack((X_p, [1]))).ravel()  # Agregar uno al bias para considerar el bias

                # Formar la matriz con las derivadas
                H = np.hstack((dw0, dw1))[np.newaxis, :]
                
                # Ganancia de Kalman
                K = P.dot(H.T).dot(np.linalg.pinv(R + H.dot(P).dot(H.T)))  # Usar pseudoinversa para evitar errores

                # Calcular delta para actualizar pesos
                dw = self.eta * K.dot(e)

                # Actualizar los pesos de ambas capas
                self.w[0] = self.w[0] + dw[:self.w[0].size].reshape(self.w[0].shape)
                self.w[1] = self.w[1] + dw[self.w[0].size:].reshape(self.w[1].shape)

                # Actualizar matriz P
                P = P - np.dot(K, H.dot(P))
                if Q.any():  # Agregar Q solo si es diferente de 0
                    P = P + Q
        print("----------------------- Fin del Entrenamiento -------------------")
                
# Definir funcion a aproximar
def f(X, s=1):
    sum2 = X[0,:]**2 + X[1, :]**2
    Z = (10 / (np.pi * s**4)) * (1 - (1/2 * ((sum2) / s**2))) * np.exp(-(sum2) / 2 * s**2)
    return Z


# Crear dataset mediante una malla y el valor de la funcion en cada putno
xx, yy = np.meshgrid(np.linspace(-5, 5, 35), np.linspace(-5, 5, 35))
X = np.array([xx.ravel(), yy.ravel()])
Y = f(X)

# Graficar la superficie
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Aproximacion de Funcion")
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(xx, yy, Y.reshape(xx.shape), cmap=plt.cm.plasma, linewidth=0.2, edgecolor='k', alpha=0.7)
ax.set_title("Funcion Original")

# Crear red y entrenarla 
net = MLP_Kalman(2, 128, activacion_oculta=tanh, eta=0.5)
net.fit(X, Y.reshape((1, -1)), r=0.1, epochs=100)

#Predecir con la red para cada punto de la malla
zz = (net.predict(X)).reshape(xx.shape)
# Mostrar funcion predicha
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(xx, yy, zz, cmap=plt.cm.plasma, linewidth=0.2, edgecolor='k', alpha=0.7)
ax.set_title("Funcion predicha")

# Graficar el error
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(xx, yy, (Y.reshape(xx.shape) - zz), cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.7)
ax.set_title(r"Error ($deseada - predicha$)")

plt.show()

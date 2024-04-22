# Clase para Red Neuronal Multicapa entrenada con filtro de Kalman
# Angel Isaac Gomez Canales

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

class MLP():
    def __init__(self, dims_capas, activacion_oculta=relu, activacion_salida=logistica, eta=0.1):
        # Instanciar atributos con listas vacias
        self.L = len(dims_capas) - 1  # Numero de capas (sin contar entrada)
        self.w = [None]  # pesos de cada capa
        self.b = [None]  # bias de cada capa
        self.f = [None]  # funcion de activacion de cada capa
        self.capas = dims_capas  # Numero de neuronas en cada capa
        self.eta = eta

        # Inicilizar los pesos y bias de cada capa aleatoriamente [-1, 1]
        for l in range(1, self.L+1):
            self.w.append(-1 + 2 * np.random.rand(self.capas[l], self.capas[l-1])) 
            self.b.append(-1 + 2 * np.random.rand(self.capas[l], 1))

            # Para cada capa se asigna su funcion de activacion dependiendo si es oculta o salida
            if l == self.L:
                self.f.append(activacion_salida)
            else:
                self.f.append(activacion_oculta)

    # Metodo para hacer predicciones
    def predict(self, X):
        A = X
        # Esta A ira pasando capa por capa, por lo que ira cambiando
        for l in range(1, self.L+1):  # Pasar las activaciones por cada capa
            A = self.f[l](self.w[l] @ A + self.b[l])
        return A
    
    # Metodo para entrenar la red
    def fit(self, X, Y, epochs=500, batch_size=1):
        # Gradiente descendente por n epocas
        for _ in range(epochs):
            # Mezclar los datos al iniciar cada epoca
            idx = np.random.permutation(X.shape[1])
            X = X[:, idx]
            Y = Y[:, idx]

            # Validar el tamano de lote
            if batch_size < 1 or batch_size > X.shape[1]:
                print(f"Elija un tamano de batch valido, debe ser en el rango [1, {X.shape[1]}]")
                break

            # Entrenar con cada lote
            for i in range(int(np.ceil(X.shape[1] / batch_size))):
                # Tomar el lote del conjunto de datos
                # Esto no genera error ya que numpy solo toma los elementos
                # dentro de los limites
                X_batch = X[:, i * batch_size: (i+1) * batch_size]
                Y_batch = Y[:, i * batch_size: (i+1) * batch_size]

                # Incializar Activaciones y sus derivadas (se usan en retropropagacion)
                As = []
                dA = [None]
                lg = [None] * (self.L+1)

                # Propagacion hacia adelante
                A = X_batch.copy()  # Copia para mantener la entrada original y no modificarla
                As.append(A)
                for l in range(1, self.L+1):
                    # Calcular activacion y derivada de cada capa y guardarlas
                    A, da = self.f[l](self.w[l] @ A + self.b[l], derivative=True)
                    As.append(A)
                    dA.append(da)

                # Retropropagacion
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        lg[l] = (Y_batch - As[l]) * dA[l]  # gradiente local capa salida
                    else:
                        lg[l] = (self.w[l+1].T @ lg[l+1]) * dA[l]  # gradiente local capa oculta

                # Actualizar pesos y bias de cada capa (regla delta)
                for l in range(1, self.L+1):
                    self.w[l] += self.eta/batch_size * (lg[l] @ As[l-1].T)
                    self.b[l] += self.eta/batch_size * np.sum(lg[l], axis=1, keepdims=True)

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
        """
        # Inicilizar los pesos y bias de cada capa aleatoriamente [-1, 1]
        self.nW = 0  # Numero de pesos
        for l in range(self.L):
            self.w.append(-1 + 2 * np.random.rand(self.capas[l+1], self.capas[l]+1))

            # Agregar el numero de pesos de la capa al total de pesos (contando el bias)
            self.nW += self.capas[l+1] * self.capas[l] + 1

            # Para cada capa se asigna su funcion de activacion dependiendo si es oculta o salida
            if l+1 == self.L:
                self.f.append(activacion_salida)
            else:
                self.f.append(activacion_oculta)
        """


    # Metodo para hacer predicciones
    def predict(self, X):
        a_h = self.f(self.w[0][:, :-1] @ X + self.w[0][:, -1:])  # Activacion de oculta
        y = self.w[1][:, :-1] @ a_h + self.w[1][:, -1:]  # Salida de la red
        return y
    
    # Metodo para entrenar la red
    def fit(self, X, Y, epochs=100, p=0.5, q=0, r=0.1):
        
        # Crear Matrices P, Q y R
        P = p * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        Q = q * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        if r == None:
            r = self.eta ** -1
        R = r * np.eye(Y.shape[0])  # Tamaño de Nsalidas x Nsalias (m)
        # Entrenar por n epocs
        for i in range(epochs):
            print(f"Iniciando Batch {i+1}")
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
                a_h, da_h = self.f(self.w[0][:, :-1] @ X_p + self.w[0][:, -1:], derivative=True)  # Activacion de oculta y derivada
                y = self.w[1][:, :-1] @ a_h + self.w[1][:, -1:]  # Salida de la red

                # Calcular el error
                e = Y_p - y

                # Obtener matriz de derivadas
                # Primero derivada de la salida con respecto a pesos de la salida (son las activaciones de la capa oculta)
                dw1 = a_h
                # Añadir la derivada con respecto al bias y aplanar a vector
                dw1 = np.vstack([dw1, 1]).flatten()
                # Las dervadas de la salida con respecto a pesos de la capa oculta (son la derivada de 
                # la activacion de la capa oculta por los pesos de la salida y esto por la entrada)
                dw0 = np.dot((self.w[1][:, :-1] * da_h.flatten()).T, X_p.T)
                # Añadir la derivada con respecto al bias y aplanar a vector
                dw0 = np.hstack((dw0, (self.w[1][:, :-1] * da_h.flatten()).T)).flatten()

                # Crear matriz H con las derivadas
                H = np.hstack((dw0, dw1)).reshape((-1, 1))

                # Matriz K
                K = P @ H @ np.linalg.pinv(R + H.T @ P @ H)  # Usar pseudoinversa para evitar errores
                # Calcular el delta para los pesos
                dw = self.eta * K @ e
                # Actualizar pesos de ambas capas
                self.w[0] += dw[:self.w[0].size].reshape(self.w[0].shape)
                self.w[1] += dw[self.w[0].size:].reshape(self.w[1].shape)

                # Actualizar matriz P
                P -= K @ H.T @ P + Q

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
net = MLP_Kalman(2, 128, activacion_oculta=tanh, eta=0.1)
net.fit(X, Y.reshape((1, -1)), epochs=100)

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

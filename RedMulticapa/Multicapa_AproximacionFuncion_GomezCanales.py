# Aproximacion de funciones con red neuronal multicapa
# Angel Isaac Gomez Canales 15/03/2024

# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt

# Definir funcion a aproximar
def f(X, s=1):
    sum2 = X[0,:]**2 + X[1, :]**2
    Z = (10 / (np.pi * s**4)) * (1 - (1/2 * ((sum2) / s**2))) * np.exp(-(sum2) / 2 * s**2)
    return Z

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

# Crear dataset mediante una malla y el valor de la funcion en cada putno
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
X = np.array([xx.ravel(), yy.ravel()])
Y = f(X)

# Graficar la superficie
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(xx, yy, Y.reshape(xx.shape), cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.7)
# Crear red y entrenarla 
net = MLP((2, 32, 16, 8, 1), activacion_oculta=tanh, activacion_salida=lineal, eta=0.5)
net.fit(X, Y.reshape((1, -1)), epochs=500, batch_size=32)

#Predecir con la red para cada punto de la malla
zz = (net.predict(X)).reshape(xx.shape)
# Mostrar funcion predicha
ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(xx, yy, zz, cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.7)

# Graficar el error
ax = fig.add_subplot(223, projection='3d')
ax.plot_surface(xx, yy, (Y.reshape(xx.shape) - zz), cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.7)

# Red vs Original
ax = fig.add_subplot(224, projection='3d')
ax.plot_surface(xx, yy, Y.reshape(xx.shape), cmap=plt.cm.viridis, linewidth=0, antialiased=False, alpha=0.5)
ax.plot_surface(xx, yy, zz, cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.3)
plt.show()

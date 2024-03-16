# Angel Isaac Gomez Canales
# Red Unicapa para clasificacion 04/03/2024

# Importar librerias requeridas 
import numpy as np
import matplotlib.pyplot as plt

def standardScaler(X):
    """
    Funcion para escalar los datos
    input: X: data
    output: datos escalados, media y desv estandar
    """
    mean = np.mean(X, axis=1).reshape(-1, 1)
    std = np.std(X, axis=1).reshape(-1, 1)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

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

class RedUniCapa:
    def __init__(self, n_input, n_output, activacion=lineal):
        # Inicializar pesos y bias aleatoriamente [0, 1]
        self.w = np.random.rand(n_output, n_input)
        self.b = np.random.rand(n_output, 1)
        self.f = activacion  # Funcion de activacion
    
    def predict(self, X):
        return self.f(np.dot(self.w, X) + self.b)
    
    def fit(self, X, Y, batch_size, epochs, eta=0.5, historia=True):
        # Lista para guardar la perdida (serviran para graficar)
        loss = [np.sqrt(np.sum((Y - self.predict(X)) ** 2) / X.shape[1])]
        for epoch in range(epochs):
            # Mezclar los datos
            idx = np.random.permutation(X.shape[1])
            X = X[:, idx]
            Y = Y[:, idx]
            # Suma del error cuadrado
            sum_e2 = 0
            # Validar el batch size
            if batch_size < 1 or batch_size > X.shape[1]:
                print(f"Elija un tamano de batch valido, debe ser en el rango [1, {X.shape[1]}]")
                break
            for i in range(int(np.ceil(X.shape[1] / batch_size))):
                # Tomar el lote del conjunto de datos
                # Esto no genera error ya que numpy solo toma los elementos
                # dentro de los limites
                X_batch = X[:, i * batch_size: (i+1) * batch_size]
                Y_batch = Y[:, i * batch_size: (i+1) * batch_size]

                # Actualizar pesos y bias
                Y_pred, dY = self.f(np.dot(self.w, X_batch) + self.b, derivative=True) # Obtener prediccion sobre el batch
                error = Y_batch - Y_pred  # Calcular el error
                e2 = error ** 2 # Error cuadrado
                lg = error * dY  # Gradiente local
                sum_e2 += np.sum(e2)
                # Actualizacion de pesos y bias
                self.w += eta / batch_size * (lg @ X_batch.T)
                self.b += eta / batch_size * np.sum(lg, axis=1).reshape(-1, 1)

            rmse = np.sqrt(sum_e2 / X.shape[1])
            loss.append(rmse)

        if historia:
            return loss

# Cargar los datos (cambiar las rutas)
X = np.loadtxt("./Practica3_RedUnicapa_GomezCanales/data.csv", delimiter=',')
Y = np.loadtxt("./Practica3_RedUnicapa_GomezCanales/deseada.csv", delimiter=',')

# Escalar los datos
X, _, _ = standardScaler(X)

# Graficar los datos con los colores
plt.figure()
plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=0), cmap='rainbow', edgecolors='k')

# Crear Red unicapa y entrenarla
net = RedUniCapa(2, 4, activacion=logistica)
net.fit(X, Y, batch_size=X.shape[1], epochs=250, historia=False)

# Crear malla para graficar area de clasificacion
xmin = np.min(X, axis=1)[0] - 0.1
xmax = np.max(X, axis=1)[0] + 0.1
ymin = np.min(X, axis=1)[1] - 0.1
ymax = np.max(X, axis=1)[1] + 0.1
xx = np.linspace(xmin, xmax, 100)
yy = np.linspace(ymin, ymax, 100)
Xx, Yy = np.meshgrid(xx, yy)
grid = np.vstack((Xx.ravel(), Yy.ravel()))

# Predecir para cada punto de la malla
Ypred_grid = np.argmax(net.predict(grid), axis=0).reshape(Xx.shape)

# Pintar el area de decision
plt.contourf(Xx, Yy, Ypred_grid, levels=np.arange(-.5, 4, 1), cmap='rainbow', alpha=0.4)
plt.title("Contorno de decision de la red")
plt.xlabel(r"x", fontsize=14)
plt.ylabel(r"y", fontsize=14)
plt.axis([xmin, xmax, ymin, ymax])
plt.grid()
plt.show()


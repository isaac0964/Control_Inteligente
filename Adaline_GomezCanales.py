# Angel Isaac Gomez Canales
# Adaline 29/02/2024

# Importar librerias requeridas 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def standardScaler(X):
    """
    Funcion para escalar los datos
    input: X: data
    output: datos escalados, media y desv estandar
    """
    mean = np.mean(X, axis=-1)
    std = np.std(X, axis=-1)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def animar_recta(i, ws, bs, x, plotl):
    """
    Esta funcion anima la recta
    input: 
        - i: iteracion actual
        - ws: pesos
        - bs: bias
        - x: linspace con los valores para calcular y=mx+b
        - plotl: plot donde se hara la animacion
    """
    plotl.set_ydata(ws[i]*x + bs[i])
    return plotl

def animar_pesos(i, ws, bs, its, plotw, plotb):
    """
    Esta funcion anima los pesos
    input: 
        - i: iteracion actual
        - ws: pesos
        - bs: bias
        - its: lista con las iteraciones
        - plotw: esqueleto de los pesos
        - plotb: esqueleto del bias
    """
    plotw.set_data(its[:i+1], ws[:i+1])
    plotb.set_data(its[:i+1], bs[:i+1])
    return (plotw, plotb)

def animar_loss(i, losses, its, plote):
    """
    Esta funcion anima los pesos
    input: 
        - i: iteracion actual
        - loss: costo de cada iteracion
        - its: lista con las iteraciones
        - plote: esqueleto del plot de lcosto
    """
    plote.set_data(its[:i+1], losses[:i+1])
    return plote

# Crear clase para Adaline
class Adaline:
    def __init__(self, n_input, n_output):
        # Inicializar pesos y bias aleatoriamente [0, 1]
        self.w = np.random.rand(n_output, n_input)
        self.b = np.random.rand(n_output, 1)
    
    def predict(self, X):
        return np.dot(self.w, X) + self.b
    
    def fit(self, X, Y, batch_size, epochs, eta=0.1, historia=True):
        # Lista para guardar pesos del algoritmo y perdida (serviran para graficar)
        wb = [(self.w[0,0], self.b[0,0])]
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
                Y_pred = self.predict(X_batch)  # Obtener prediccion sobre el batch
                error = Y_batch - Y_pred  # Calcular el error
                e2 = error ** 2 # Error cuadrado
                sum_e2 += np.sum(e2)
                # Actualizacion de pesos y bias
                self.w += eta / batch_size * (error @ X_batch.T)
                self.b += eta / batch_size * np.sum(error, axis=1)

            rmse = np.sqrt(sum_e2 / X.shape[1])
            # Guardar pesos y perdoda
            wb.append((self.w[0, 0], self.b[0,0]))
            loss.append(rmse)

        if historia:
            return wb, loss

# Datos (problema de regresion)
X = np.array([16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8, 17.83, 79.24, 21.50, 40.33, 21, 13.50, 
              17.74, 24, 29, 15.35, 19, 9.50, 35.10, 17.90, 52.32, 18.75, 19.83, 10.75]).reshape(1, -1)  # Reshape para darle forma adecuada (m, p)
Y = np.array([7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6, 7, 3, 17, 10, 26, 9, 8, 4]).reshape(1, -1)  # Reshape para darle forma adecuada (n, p)

# Escalar los datos
X_scaled, _, _ = standardScaler(X)
Y_scaled, _, _ = standardScaler(Y)

# Entrenar Adaline
net = Adaline(1, 1)
tam_batch = 1  # Tamano de lotes
epocas = 250
wb, loss = net.fit(X_scaled, Y_scaled, tam_batch, epocas, eta=0.05, historia=True)

# Separar pesos y bias
ws, bs = list(zip(*wb))
# lista con las iteraciones
its = list(range(len(ws)))

# Visualizacion de datos
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
fig.suptitle("Evolucion de la Red")

# Scatter de los datos y grfica de la linea
plot_l, = ax[0].plot(X_scaled[0, :], Y_scaled[0, :], 'k.')
ax[0].set_title("Datos y regresion")
ax[0].set_xlabel(r"$x$", fontsize=16)
ax[0].set_ylabel(r"$y$", fontsize=16)
x = np.linspace(-1.2, 4, 100)
plotl, =  ax[0].plot(x, ws[0] * x + bs[0], "r--")
anim_recta = animation.FuncAnimation(fig=fig, func=animar_recta, frames=len(ws), fargs=(ws, bs, x, plotl), interval=10, repeat=False)

# Graficar pesos y bias
ax[1].set_title("Pesos y bias")
ax[1].set_xlabel("Epoca", fontsize=14)
ax[1].axis([0, len(ws), -0.25, 1.25])
plotw, = ax[1].plot([], [], "mo-", label="w", markevery=[-1])
plotb, = ax[1].plot([], [], "go-", label="b", markevery=[-1])
ax[1].legend(loc="upper right", fontsize="large")
anim_wb = animation.FuncAnimation(fig=fig, func=animar_pesos, frames=len(ws), fargs=(ws, bs, its, plotw, plotb), interval=10, repeat=False)

# Graficar el costo
ax[2].set_title("Costo (RMSE)")
ax[2].set_xlabel("Epoca", fontsize=14)
ax[2].set_ylabel("RMSE", fontsize=14)
ax[2].axis([0, len(ws), 0, 2])
plote, = ax[2].plot([], [], "ro-", markevery=[-1])
anim_loss = animation.FuncAnimation(fig=fig, func=animar_loss, frames=len(ws), fargs=(loss, its, plote), interval=10, repeat=False)
plt.tight_layout()
plt.show()
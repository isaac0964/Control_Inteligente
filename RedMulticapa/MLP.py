# Clase para Red Neuronal Multicapa
# Angel Isaac Gomez Canales 15/03/2024

# Importar librerias requeridas
import numpy as np

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
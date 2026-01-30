import numpy as np

class Neurona:
    # Constructor de la clase Neurona
    def __init__(self, pesos: list, sesgo, funcion_activacion='sigmoide'):
        self.pesos = pesos
        self.sesgo = sesgo

        # Diccionario de funciones de activacion
        funciones_activacion = {
            'sigmoide': Neurona.__funcion_sigmoide,
            'relu': Neurona.__funcion_relu,
            'tanh': Neurona.__funcion_tanh,
            'binary_step': Neurona.__funcion_binary_step
        }

        self.funcion_activacion = funciones_activacion[funcion_activacion]

    # Logica de la funcion de activacion
    # Sigmoide
    @staticmethod
    def __funcion_sigmoide(x):
        return 1 / (1 + np.exp(-x))
    
    # ReLU (Rectified Linear Unit)
    @staticmethod
    def __funcion_relu(x):
        return max(0, x)
    
    # tangente hiperbólica
    @staticmethod
    def __funcion_tanh(x):
        return np.tanh(x)
    
    # binary step function
    @staticmethod
    def __funcion_binary_step(x):
        return 1 if x >= 0 else 0

    # Método para activar la neurona
    def run(self, entradas):
        y = np.dot(entradas, self.pesos) + self.sesgo
        return self.funcion_activacion(y)

    # Cambiar sesgo
    def set_sesgo(self, nuevo_sesgo):
        self.sesgo = nuevo_sesgo

    # Cambiar pesos
    def set_pesos(self, nuevos_pesos):
        self.pesos = nuevos_pesos
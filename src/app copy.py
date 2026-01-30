import streamlit as st
import numpy as np
from Neurona import Neurona

st.title('Entrenamiento neuronal')
st.image("data/neurona.webp", width=400)
#st.image("src/data/neurona.webp", width=400)
st.write("Una neurona es una unidad básica de procesamiento en una red neuronal artificial. Recibe entradas, las procesa y produce una salida basada en los pesos asignados a cada entrada.")
st.write("En este ejemplo entrenamos una neurona para conseguir los pesos y sesgo exactos para obtener unas salidas concretas. El valor predefinido es para conseguir un y_logico")


#"""
#Pesos y sesgo iniciales aleatorios
#
#Calcular salida
#Calcular error:
#e = salida_esperada - salida_obtenida
#si error != 0:
#
#peso nuevo = velocidad de aprendizaje * error * entrada
#sesgo nuevo = velocidad de aprendizaje * error
#
#Ajustar pesos y sesgo
#Repetir hasta que el error sea -1%
#
#Entradas y salidas esperadas:
#0 0 -> 0
#0 1 -> 0
#1 0 -> 0
#1 1 -> 1
#
#Usar binary step como funcion de activacion
#"""
entradas = [[0,0], [0,1], [1,0], [1,1]]
salidas_esperadas = [0, 0, 0, 1]
pesos = [np.random.random(), np.random.random()]
sesgo = np.random.random()
tasa_aprendizaje = 0.2

def entrenar_perceptron(entradas, salidas_esperadas, pesos, sesgo, tasa_aprendizaje, max_iter=1000):
    for n in range(len(entradas)):
        for _ in range(max_iter):
            salida = Neurona(pesos, sesgo, 'binary_step').run(entradas[n])
            error = salidas_esperadas[n] - salida
            if error != 0:
                pesos[0] += tasa_aprendizaje * error * entradas[n][0]
                pesos[1] += tasa_aprendizaje * error * entradas[n][1]
                sesgo += tasa_aprendizaje * error
            
            #st.write(f"Iteración {_+1} para entrada {entradas[n]}: Salida={salida}, Error={error}, Pesos={pesos}, Sesgo={sesgo}")
            if _ == max_iter-1:
                st.write(f"Iteración {_+1} para entrada {entradas[n]}: Salida={salida}, Error={error}, Pesos={pesos}, Sesgo={sesgo}")

    return pesos, sesgo

st.write("Entrenando perceptron para Y lógico...")
st.write(f"Pesos iniciales: {pesos}, Sesgo inicial: {sesgo}")
pesos_entrenados, sesgo_entrenado = entrenar_perceptron(entradas, salidas_esperadas, pesos, sesgo, tasa_aprendizaje)

st.write(f"Pesos entrenados: {pesos_entrenados}, Sesgo entrenado: {sesgo_entrenado}")


st.subheader("Verificación del Y lógico entrenado")
neuron_y_logico = Neurona(pesos_entrenados, sesgo_entrenado, 'binary_step')
for i in range(len(entradas)):
    salida = neuron_y_logico.run(entradas[i])
    st.write(f"Entrada: {entradas[i]} -> Salida: {salida} (Esperada: {salidas_esperadas[i]})")  



st.write("© Cristina Vacas López - CPIFP Alan Turing")
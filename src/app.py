import streamlit as st
import numpy as np
import pandas as pd
from Neurona import Neurona

### Funciones ###

def entrenar_perceptron(entradas, salidas_esperadas, pesos_inicales, sesgo_inicial, tasa_aprendizaje, funcion_activacion, max_iter=1000):
    pesos = pesos_inicales
    sesgo = sesgo_inicial
    for n in range(len(entradas)):
        for _ in range(max_iter):
            salida = Neurona(pesos, sesgo, funcion_activacion).run(entradas[n])
            error = salidas_esperadas[n] - salida
            if error != 0:
                pesos[0] += tasa_aprendizaje * error * entradas[n][0]
                pesos[1] += tasa_aprendizaje * error * entradas[n][1]
                sesgo += tasa_aprendizaje * error
            
            #if _ == max_iter-1:
            #    st.write(f"Iteración {_+1} para entrada {entradas[n]}: Salida={salida}, Error={error}, Pesos={pesos}, Sesgo={sesgo}")

    return pesos, sesgo

### Interfaz ###

st.title('Entrenamiento neuronal')
#st.image("data/neurona.webp", width=400) # Img para despliegue en local / docker
st.image("src/data/neurona.webp", width=400) # Img para despliegue en Streamlit.io

st.write("Esta es una aplicación educativa desarrollada con Streamlit que permite a los estudiantes comprender visualmente cómo funciona el entrenamiento de una neurona artificial mediante el algoritmo del perceptrón.")
st.write("El objetivo es simular y demostrar el proceso de aprendizaje de una neurona, mostrando cómo se ajustan los pesos y sesgos a través de iteraciones sucesivas para clasificar correctamente datos de entrada.")

tab1, tab2, tab3 = st.tabs(["Entrenamiento Neuronal", "Ejemplo Y Logico", "Ejemplo O Logico"])

with tab1:
    cols = st.columns(2) 
    n_datos = cols[0].slider('Datos de entrenamiento', 1, 10, 4)
    n_entradas = cols[1].slider('Entradas por dato', 1, 8, 2)
    
    entradas = []
    salidas_esperadas = []

    for d in range(n_datos):
        st.write(f"Datos de entrenamiento {d+1}")
        cols = st.columns(int(n_entradas) + 1) 
    
        entrada_actual = []
        for e in range(int(n_entradas)):
            with cols[e]:
                valor = st.number_input(f"Entrada {e+1}", key=f"d{d}_e{e}", value=0.0)
                entrada_actual.append(valor)

        with cols[-1]:
            salida_actual = st.number_input(f"Salida {d+1}", key=f"salida_{d}", value=0.0)

        entradas.append(entrada_actual)
        salidas_esperadas.append(salida_actual)
    
    pesos = [np.random.random() for e in range(n_entradas)]
    sesgo = np.random.random()

    cols = st.columns(2) 
    tasa_aprendizaje = cols[0].slider("Tasa de aprendizaje", 0.01, 1.0, 0.2)

    options = {
        'Binary Step': 'binary_step',
        'Sigmoide': "sigmoide",
        'ReLu': 'relu',
        'Tangente Hiperbólica': 'tanh',
    }

    funcion_activacion = cols[1].selectbox("Selecciona la función de activación:", options=options, key="funcion_activacion")


    if st.button("Entrenar perceptron"):
        st.write(f"Pesos iniciales: {pesos}, Sesgo inicial: {sesgo}")
        notificacion = st.info("Entrenando perceptron...")

        # Entrenamiento
        pesos_entrenados, sesgo_entrenado = entrenar_perceptron(entradas, salidas_esperadas, pesos, sesgo, tasa_aprendizaje, options[funcion_activacion])

        if pesos_entrenados and sesgo_entrenado:
            notificacion = st.success(f"Entrenamiento finalizado satisfactoriamente:\nPesos entrenados: {pesos_entrenados}.  Sesgo entrenado: {sesgo_entrenado}")

            st.subheader("Verificación del Y lógico entrenado")
            perceptron = Neurona(pesos_entrenados, sesgo_entrenado, options[funcion_activacion])
            for i in range(len(entradas)):
                salida = perceptron.run(entradas[i])
                st.write(f"Entrada: {entradas[i]} -> Salida: {salida} (Esperada: {salidas_esperadas[i]})")  

with tab2:
    st.subheader("Datos de entrenamiento para Puerta Lógica AND (Y)")

    # Variables para el entrenamiento neuronal de un Y Logico
    entradas = [[0,0], [0,1], [1,0], [1,1]]
    salidas_esperadas = [0, 0, 0, 1]
    pesos = [np.random.random(), np.random.random()]
    sesgo = np.random.random()
    tasa_aprendizaje = 0.2
    funcion_activacion = 'binary_step'

    # Tabla de  datos
    df_cols = {f"Entrada {i+1}": [row[i] for row in entradas] for i in range(int(n_entradas))}
    df_cols["Salida Esperada"] = salidas_esperadas
    df_entr = pd.DataFrame(df_cols)
    st.subheader("Tabla de datos de entrenamiento")
    st.dataframe(df_entr, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Tasa de aprendizaje:** {tasa_aprendizaje}")
    with col2:
        st.info(f"**Funcion de activacion** {funcion_activacion}")
    

    if st.button("🚀 Entrenar OR", use_container_width=True, type="primary", key="button-AND"):

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Pesos Iniciales:** {[f'{p:.4f}' for p in pesos]}")
        with col2:
            st.info(f"**Sesgo Inicial:** {sesgo:.4f}")


        with st.spinner("⏳ Entrenando puerta OR..."):
            pesos_entrenados, sesgo_entrenado = entrenar_perceptron(entradas, salidas_esperadas, pesos, sesgo, tasa_aprendizaje, funcion_activacion)
            
            st.subheader("Resultados de pesos y sesgo")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Pesos Entrenados:** {[f'{p:.4f}' for p in pesos_entrenados]}")
            with col2:
                st.success(f"**Sesgo Entrenado:** {sesgo_entrenado:.4f}")

            # Verificacion del perceptron entrenado
            st.subheader("Verificación")
            st.write("Si no obtienes los resultados esperados vuelve a entrenar hasta hayar los pesos y el sesgo que obtengan los resultados esperados.")
            neuron_or = Neurona(pesos_entrenados, sesgo_entrenado, 'binary_step')
            
            resultados = []
            for i in range(len(entradas)):
                salida = neuron_or.run(entradas[i])
                es_correcto = "✅" if abs(salida - salidas_esperadas[i]) < 0.5 else "❌"
                resultados.append({
                    "Entrada": f"{entradas[i][0]} OR {entradas[i][1]}",
                    "Salida": int(salida),
                    "Salida Esperada": int(salidas_esperadas[i]),
                    "Resultado": es_correcto
                })
            
            st.dataframe(resultados, use_container_width=True)
            st.write(f"Una vez obtenido los resultados esperados ya puede probar su neurona con dichos pesos y sesgo en el {st.page_link("https://simulador-neuronal-vlc.streamlit.app/", label="Simulador neuronal")}, recuerde poner la funcion de activacion correspondiente.")

        
with tab3:
    st.subheader("Ejemplo de entrenamiento para Puerta Lógica OR (O)")

    # Variables para el entrenamiento neuronal de un O Logico
    entradas = [[0,0], [0,1], [1,0], [1,1]]
    salidas_esperadas = [0, 1, 1, 1]
    pesos = [np.random.random(), np.random.random()]
    sesgo = np.random.random()
    tasa_aprendizaje = 0.2
    funcion_activacion = 'binary_step'

    # Tabla de  datos
    df_cols = {f"Entrada {i+1}": [row[i] for row in entradas] for i in range(int(n_entradas))}
    df_cols["Salida Esperada"] = salidas_esperadas
    df_entr = pd.DataFrame(df_cols)
    st.subheader("Tabla de datos de entrenamiento")
    st.dataframe(df_entr, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Tasa de aprendizaje:** {tasa_aprendizaje}")
    with col2:
        st.info(f"**Funcion de activacion** {funcion_activacion}")
    

    if st.button("🚀 Entrenar OR", use_container_width=True, type="primary", key="button-OR"):

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Pesos Iniciales:** {[f'{p:.4f}' for p in pesos]}")
        with col2:
            st.info(f"**Sesgo Inicial:** {sesgo:.4f}")

        with st.spinner("⏳ Entrenando puerta OR..."):
            pesos_entrenados, sesgo_entrenado = entrenar_perceptron(entradas, salidas_esperadas, pesos, sesgo, tasa_aprendizaje, funcion_activacion)
            
            st.subheader("Resultados de pesos y sesgo")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Pesos Entrenados:** {[f'{p:.4f}' for p in pesos_entrenados]}")
            with col2:
                st.success(f"**Sesgo Entrenado:** {sesgo_entrenado:.4f}")

            # Verificacion del perceptron entrenado

            st.subheader("Verificación")
            st.write("Si no obtienes los resultados esperados vuelve a entrenar hasta hayar los pesos y el sesgo que obtengan los resultados esperados.")

            neuron_or = Neurona(pesos_entrenados, sesgo_entrenado, 'binary_step')
            
            resultados = []
            for i in range(len(entradas)):
                salida = neuron_or.run(entradas[i])
                es_correcto = "✅" if abs(salida - salidas_esperadas[i]) < 0.5 else "❌"
                resultados.append({
                    "Entrada": f"{entradas[i][0]} OR {entradas[i][1]}",
                    "Salida": int(salida),
                    "Salida Esperada": int(salidas_esperadas[i]),
                    "Resultado": es_correcto
                })
            
            st.dataframe(resultados, use_container_width=True)
            st.write(f"Una vez obtenido los resultados esperados ya puede probar su neurona con dichos pesos y sesgo en el {st.page_link("https://simulador-neuronal-vlc.streamlit.app/", label="Simulador neuronal")}, recuerde poner la funcion de activacion correspondiente.")


st.write("© Cristina Vacas López - CPIFP Alan Turing")




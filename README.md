# Entrenamiento Neuronal - Simulador Interactivo

Aplicación educativa desarrollada con **Streamlit** que permite comprender visualmente cómo funciona el entrenamiento de una neurona artificial mediante el algoritmo del perceptrón.

## 📚 Descripción

Esta herramienta educativa simula el proceso de aprendizaje de una neurona, mostrando cómo se ajustan los **pesos y sesgos** a través de iteraciones sucesivas para clasificar correctamente datos de entrada. Es ideal para estudiantes que desean entender conceptos fundamentales de redes neuronales.

## ✨ Características Principales

- **Entrenamiento Automático**: Implementa el algoritmo del perceptrón con ajuste dinámico de pesos y sesgos
- **Interfaz Interactiva**: Configuración flexible de datos de entrenamiento y parámetros
- **Múltiples Funciones de Activación**: Binary Step, Sigmoide, ReLu, Tangente Hiperbólica
- **Ejemplos Predefinidos**: Incluye demostraciones de puertas lógicas AND (Y) y OR (O)
- **Visualización del Proceso**: Muestra pesos iniciales, finales, sesgo y predicciones

## 🚀 Instalación

**Clona o descarga el proyecto**

**Ejecutar con Docker**

Usar Docker para evitar problemas de dependencias:

```bash
docker-compose up
```

La aplicación estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
entrenamiento-neuronal/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias de Python
├── Dockerfile               # Configuración de Docker
├── docker-compose.yml       # Orquestación de contenedores
└── src/
    ├── app.py              # Aplicación principal con interfaz Streamlit
    ├── Neurona.py          # Clase de la neurona artificial
    └── data/
        └── neurona.webp    # Imagen educativa de una neurona
```

## 🎮 Cómo Usar

### Pestaña 1: Entrenamiento Neuronal
1. Configura el número de **datos de entrenamiento** (1-10)
2. Define el número de **entradas por dato** (1-8)
3. Ingresa los valores de entrada y salida esperada para cada dato
4. Ajusta la **tasa de aprendizaje** (0.01-1.0)
5. Selecciona la **función de activación** deseada
6. Haz clic en **"Entrenar perceptron"**
7. Visualiza los pesos y sesgo entrenados, así como la verificación de predicciones

### Pestaña 2: Ejemplo Y Lógico (AND)
Demuestra cómo entrenar una neurona para implementar la puerta lógica AND:
- Entrada: [0,0] → Salida: 0
- Entrada: [0,1] → Salida: 0
- Entrada: [1,0] → Salida: 0
- Entrada: [1,1] → Salida: 1

### Pestaña 3: Ejemplo O Lógico (OR)
Demuestra cómo entrenar una neurona para implementar la puerta lógica OR:
- Entrada: [0,0] → Salida: 0
- Entrada: [0,1] → Salida: 1
- Entrada: [1,0] → Salida: 1
- Entrada: [1,1] → Salida: 1

## 📊 Parámetros Configurables

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| Datos de Entrenamiento | 1-10 | Número de ejemplos para entrenar |
| Entradas por Dato | 1-8 | Dimensionalidad de entrada |
| Tasa de Aprendizaje | 0.01-1.0 | Velocidad de convergencia del modelo |
| Función de Activación | 4 opciones | Define el comportamiento de la neurona |
| Iteraciones Máximas | 1000 | Límite de ciclos de entrenamiento |

## 🧠 Conceptos Educativos

Esta aplicación enseña:
- **Propagación hacia adelante** (Forward Pass)
- **Cálculo de errores** de predicción
- **Ajuste de pesos y sesgos** mediante el algoritmo del perceptrón
- **Funciones de activación** y su impacto en el comportamiento neuronal
- **Tasa de aprendizaje** y convergencia

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para crear aplicaciones web interactivas
- **NumPy**: Computación numérica y manejo de arrays
- **Python 3**: Lenguaje de programación

## 📋 Requisitos (requirements.txt)

```
streamlit
numpy
```

## 🤝 Autor

© Cristina Vacas López - CPIFP Alan Turing

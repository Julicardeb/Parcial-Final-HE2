# Plataforma de Evaluación Financiera

Esta aplicación proporciona dos funcionalidades principales: cálculo del puntaje de crédito y recomendaciones de inversión basadas en el perfil de riesgo. Está diseñada para personas interesadas en mejorar sus decisiones financieras.

## Características principales

1. **Puntaje de Crédito**: 
   - Calcula el puntaje de crédito basado en información demográfica y financiera.
   - Genera recomendaciones personalizadas para mejorar o mantener el puntaje.

2. **Recomendaciones de Inversión**:
   - Evalúa el perfil de riesgo financiero del usuario.
   - Genera asignaciones sugeridas de inversión basadas en el monto especificado.
   - Proporciona gráficos y datos de mercado en tiempo real.

---

## Tecnologías utilizadas

- **Python**: Lenguaje principal.
- **Bibliotecas**:
  - `pandas`, `scikit-learn`: Procesamiento de datos y aprendizaje automático.
  - `imbalanced-learn (SMOTE)`: Manejo de datos desbalanceados.
  - `yfinance`: Datos de mercado en tiempo real.
  - `gradio`: Creación de la interfaz interactiva.
  - `transformers` y `torch`: Modelos preentrenados para generación de texto.
  - `matplotlib`: Generación de gráficos.

---

Instale los paquetes requeridos con:
```bash
    pip install -r requirements.txt
```

## Configuración:

Clona el repositorio:
```bash
git clone https://github.com/Julicardeb/Parcial-Final-HE2.git
```

## Ejemplo de uso:

1. **Evaluación del Puntaje de Crédito**: 

    Este módulo toma información básica del usuario, como edad, ingresos y nivel educativo, para predecir su nivel de puntaje de crédito y proporcionar recomendaciones específicas.

    ### Entrada
    - Edad
    - Genero
    - Ingresos mensuales (en pesos colombianos)
    - Nivel educativo
    - Estado civil
    - Número de hijos
    - Tipo de propiedad de vivienda

    ### Salida:
    - Predicción del puntaje de crédito (`Bajo`, `Promedio`, `Alto`).
    - Recomendaciones específicas para mejorar o mantener el puntaje.

2. **Perfil de Riesgo de Inversión**:
    Determina el perfil de riesgo del usuario basándose en un cuestionario. Los perfiles posibles son:
    - Conservador
    - Moderado
    - Agresivo

    ### Entrada
    - Respuestas a 7 preguntas relacionadas con la tolerancia al riesgo.
    - Monto a invertir.

    ### Salida:
    - Perfil de riesgo.
    - Recomendaciones de asignación de inversión.
    - Sugerencias de acciones e instrumentos.
    - Distribución gráfica de inversión.

## Detalles Técnicos:

1. **Preprocesamiento de Datos**: 
   - Codificación de variables categóricas mediante `LabelEncoder` y `OneHotEncoder`.
   - Escalado de características numéricas con `StandardScaler`.
   - Balanceo de clases en el conjunto de datos usando SMOTE para mejorar la precisión del modelo.

2. **Modelo de Clasificación**:
   - Algoritmo: **Random Forest**.
   - Métricas evaluadas: informe de clasificación y matriz de confusión.

3. **Generación de Recomendaciones**:
   - Traducción de datos y resultados.
   - Uso de un modelo preentrenado de generación de texto (`GPT-2 Small Spanish`) para recomendaciones en lenguaje natural.

4. **Generación de Gráficos**:
   - Visualización de asignaciones de inversión con gráficos circulares usando Matplotlib.

## Ejemplo de Resultados:

### Puntaje de Crédito:

#### Entrada:
```python
predict_credit_score(edad=30, genero='Masculino', ingresos_mensuales_pesos=5000000, educacion='Licenciatura', estado_civil='Casado/a', num_hijos=1, propiedad_vivienda='Propietario')
```

#### Salida:
```python
Su nivel de puntaje de crédito estimado es: Promedio

Recomendación:
- Mantenga un historial de pagos puntuales.
- Mantenga baja la utilización de su crédito.
- Diversifique sus tipos de crédito de manera responsable.
```

### Perfil de Riesgo de Inversión:

#### Entrada: 
Respuestas de la encuesta con un monto de $5000

#### Salida:
```python
Perfil: Moderado.
- Asigna: $2500.00 a Bonos.
- Asigna: $1500.00 a Acciones.
- Asigna: $1000.00 a Fondos Mixtos.

Acciones sugeridas: ETF SPDR S&P 500, Bonos corporativos Apple.
```
Gráfico circular generado como "grafico.png".

## Despliegue en HuggingFace Spaces :

https://huggingface.co/spaces/Julicardeb/Final_HE2

## Contribuciones:

¡Se aceptan contribuciones! Si deseas agregar funcionalidades o mejorar el sistema, no dudes en enviar un Pull Request.

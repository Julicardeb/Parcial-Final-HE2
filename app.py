from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv("Credit Score Classification Dataset.csv")
df_for_gradio = df.copy()  # For dropdowns
le = LabelEncoder()
df['Credit Score'] = le.fit_transform(df['Credit Score'])
categorical_cols = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
# Handle unknown values during prediction
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(df[categorical_cols])  # Fit before applying SMOTE and splitting

# Transform using fitted ohe
encoded_features = pd.DataFrame(ohe.transform(df[categorical_cols]))
encoded_features.columns = ohe.get_feature_names_out(categorical_cols)
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_features], axis=1)
numerical_cols = ['Age', 'Income', 'Number of Children']
scaler = StandardScaler()
scaler.fit(df[numerical_cols])

scaled_numerical = pd.DataFrame(scaler.transform(
    df[numerical_cols]))  # Use .transform()
scaled_numerical.columns = numerical_cols  # Correct column names
df = df.drop(numerical_cols, axis=1)
df = pd.concat([df, scaled_numerical], axis=1)
X = df.drop('Credit Score', axis=1)
y = df['Credit Score']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)  # Handle imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
translation_dict = {  # For the input
    # Spanish to English
    'Gender': {'Male': 'Masculino', 'Female': 'Femenino'},
    'Education': {"Bachelor's Degree": 'Licenciatura', "Master's Degree": 'Maestría', 'Doctorate': 'Doctorado', 'High School Diploma': 'Diploma de Bachillerato', "Associate's Degree": 'Grado Asociado'},
    'Marital Status': {'Single': 'Soltero/a', 'Married': 'Casado/a'},
    'Home Ownership': {'Owned': 'Propietario', 'Rented': 'Alquilado'}
}
credit_score_translation = {  # For output
    'High': 'Alto',
    'Average': 'Promedio',
    'Low': 'Bajo'
}
pesos_to_usd = 1.0 / 4200  # Pesos to USD exchange rate
months_in_year = 12
for column, translations in translation_dict.items():
    df_for_gradio[column] = df_for_gradio[column].map(
        translations).fillna(df_for_gradio[column])


def generate_recommendation(prediction_es):
    if prediction_es == 'Bajo':
        recommendation = (
            "Para mejorar su puntaje de crédito, considere las siguientes acciones:\n"
            "- Pague sus deudas a tiempo.\n"
            "- Reduzca el uso de crédito al mínimo necesario.\n"
            "- Revise su informe crediticio para corregir errores.\n"
            "- Evite abrir varias cuentas de crédito en poco tiempo."
        )
    elif prediction_es == 'Promedio':
        recommendation = (
            "Para mantener y mejorar su puntaje de crédito, le recomendamos:\n"
            "- Mantenga un historial de pagos puntuales.\n"
            "- Mantenga baja la utilización de su crédito.\n"
            "- Diversifique sus tipos de crédito de manera responsable."
        )
    elif prediction_es == 'Alto':
        recommendation = (
            "¡Felicidades por su excelente puntaje de crédito!\n"
            "- Continúe con sus buenos hábitos financieros.\n"
            "- Revise periódicamente su informe crediticio.\n"
            "- Planifique a largo plazo para mantener su estabilidad financiera."
        )
    else:
        recommendation = "No se pudo generar una recomendación específica."
    return recommendation


def predict_credit_score(edad, genero, ingresos_mensuales_pesos, educacion, estado_civil, num_hijos, propiedad_vivienda):
    try:
        # 1. Translate Input
        gender_en = translation_dict['Gender'].get(genero, genero)
        education_en = translation_dict['Education'].get(educacion, educacion)
        marital_status_en = translation_dict['Marital Status'].get(
            estado_civil, estado_civil)
        home_ownership_en = translation_dict['Home Ownership'].get(
            propiedad_vivienda, propiedad_vivienda)

        # 2. Convert Income using PPP
        ppp_conversion_factor = 1362.01  # COP per international dollar for 2021
        ingresos_anuales_ppp_dollars = ingresos_mensuales_pesos * \
            months_in_year / ppp_conversion_factor

        # 3. Create DataFrame (using English values and original column names)
        input_data = pd.DataFrame({
            'Age': [edad],
            'Gender': [gender_en],
            'Income': [ingresos_anuales_ppp_dollars],
            'Education': [education_en],
            'Marital Status': [marital_status_en],
            'Number of Children': [num_hijos],
            'Home Ownership': [home_ownership_en]
        })

        # 4. Preprocess input data
        # One-hot encode using the same ohe transformer
        encoded_features = pd.DataFrame(
            ohe.transform(input_data[categorical_cols]),
            columns=ohe.get_feature_names_out(categorical_cols)
        )
        input_data = input_data.drop(categorical_cols, axis=1)
        input_data = pd.concat([input_data, encoded_features], axis=1)

        # Scale numerical features
        scaled_numerical = pd.DataFrame(
            scaler.transform(input_data[numerical_cols]),
            columns=numerical_cols
        )
        input_data[numerical_cols] = scaled_numerical[numerical_cols]

        # Ensure the columns are in the same order as X_train.columns
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

        # 5. Predict
        prediction_numeric = model.predict(input_data)[0]

        # 6. Inverse Transform
        prediction_en = le.inverse_transform([prediction_numeric])[0]
        prediction_es = credit_score_translation.get(
            prediction_en, prediction_en)

        recommendation = generate_recommendation(prediction_es)

        # Formatear el resultado
        result = f"Su nivel de puntaje de crédito estimado es: {prediction_es}\n\nRecomendación:\n{recommendation}"

        return result

    except Exception as e:
        return f"Error en la predicción: {str(e)}"

# !pip install transformers huggingface_hub
# !pip install -U bitsandbytes
# !pip install yfinance
# !pip install matplotlib


# Recomendaciones predefinidas para perfiles
acciones_conservador = ["Bonos del Tesoro de EE.UU.",
                        "Fondo de Renta Fija Vanguard"]
acciones_moderado = ["ETF SPDR S&P 500",
                     "Bonos corporativos Apple", "Acciones Coca-Cola"]
acciones_agresivo = ["Acciones Tesla", "ETF ARK Innovation",
                     "Criptomonedas como Bitcoin o Ethereum"]

# Lista de acciones para recomendaciones dinámicas
acciones_reales = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]

# Función para calcular el perfil de riesgo basado en las respuestas


def calcular_perfil(pregunta_1, pregunta_2, pregunta_3, pregunta_4, pregunta_5, pregunta_6, pregunta_7):
    puntuacion = 0

    # Sumar puntuaciones de cada respuesta
    opciones = [pregunta_1, pregunta_2, pregunta_3,
                pregunta_4, pregunta_5, pregunta_6, pregunta_7]
    for respuesta in opciones:
        if respuesta.startswith("a."):
            puntuacion += 1
        elif respuesta.startswith("b."):
            puntuacion += 2
        elif respuesta.startswith("c."):
            puntuacion += 3

    # Determinar perfil según la puntuación acumulada
    if 9 <= puntuacion <= 14:  # Conservador
        return "Conservador"
    elif 15 <= puntuacion <= 21:  # Moderado
        return "Moderado"
    elif 22 <= puntuacion <= 27:  # Agresivo
        return "Agresivo"
    else:
        return "No determinado"

# Función para generar gráficos


def generar_grafico(asignaciones):
    categorias = list(asignaciones.keys())
    valores = list(asignaciones.values())

    plt.figure(figsize=(6, 6))
    plt.pie(valores, labels=categorias, autopct="%1.1f%%",
            startangle=140, colors=["#4CAF50", "#FFC107", "#2196F3"])
    plt.title("Distribución recomendada de inversión")
    plt.savefig("grafico.png")
    plt.close()

# Función para obtener precios de acciones en tiempo real


def obtener_precio_mercado():
    recomendaciones = {}
    for ticker in acciones_reales:
        try:
            data = yf.Ticker(ticker)
            precio = data.history(period="1d")['Close'].iloc[-1]
            recomendaciones[ticker] = f"${precio:.2f}"
        except Exception as e:
            recomendaciones[ticker] = "No disponible"
    return recomendaciones

# Función principal para recomendar inversión


def recomendar_inversion(pregunta_1, pregunta_2, pregunta_3, pregunta_4, pregunta_5, pregunta_6, pregunta_7, monto):
    perfil = calcular_perfil(pregunta_1, pregunta_2, pregunta_3,
                             pregunta_4, pregunta_5, pregunta_6, pregunta_7)
    monto = float(monto)

    if perfil == "Conservador":
        asignaciones = {"Bonos": monto * 0.8,
                        "Acciones": monto * 0.1, "Fondos Mixtos": monto * 0.1}
        recomendaciones = random.sample(acciones_conservador, 2)
    elif perfil == "Moderado":
        asignaciones = {"Bonos": monto * 0.5,
                        "Acciones": monto * 0.3, "Fondos Mixtos": monto * 0.2}
        recomendaciones = random.sample(acciones_moderado, 2)
    else:  # Agresivo
        asignaciones = {"Bonos": monto * 0.2,
                        "Acciones": monto * 0.7, "Fondos Mixtos": monto * 0.1}
        recomendaciones = random.sample(acciones_agresivo, 2)

    # Obtener precios dinámicos de mercado
    precios_mercado = obtener_precio_mercado()

    # Crear el gráfico
    generar_grafico(asignaciones)

    # Recomendación textual
    recomendacion = f"Perfil: {perfil}.\n"
    for categoria, valor in asignaciones.items():
        recomendacion += f"Asigna: ${valor:.2f} a {categoria}.\n"
    recomendacion += "\nAcciones o instrumentos sugeridos para este perfil:\n"
    recomendacion += "\n".join(f"- {rec}" for rec in recomendaciones)
    recomendacion += "\n\nPrecios actuales de acciones populares:\n"
    for ticker, precio in precios_mercado.items():
        recomendacion += f"- {ticker}: {precio}\n"

    return recomendacion, "grafico.png"


# Definir la interfaz de Gradio con pestañas
with gr.Blocks() as demo:
    gr.Markdown("# Plataforma de Evaluación Financiera")

    with gr.Tab("Puntaje de Crédito"):
        gr.Markdown("## Calculadora de Puntaje de Crédito")

        with gr.Row():
            with gr.Column():
                edad_credit = gr.Number(label="Edad", value=30)
                genero_credit = gr.Dropdown(
                    choices=['Masculino', 'Femenino'], label="Género")
                ingresos_mensuales_pesos_credit = gr.Number(
                    label="Ingresos Mensuales (Pesos Colombianos)", value=5000000)
                educacion_credit = gr.Dropdown(choices=[
                                               'Licenciatura', 'Maestría', 'Doctorado', 'Diploma de Bachillerato', 'Grado Asociado'], label="Educación")
                estado_civil_credit = gr.Dropdown(
                    choices=['Soltero/a', 'Casado/a'], label="Estado Civil")
                num_hijos_credit = gr.Number(label="Número de Hijos", value=0)
                propiedad_vivienda_credit = gr.Dropdown(
                    choices=['Propietario', 'Alquilado'], label="Propiedad de Vivienda")

        btn_credit = gr.Button("Calcular Puntaje de Crédito")
        output_credit = gr.Textbox(label="Resultado", lines=10)

        btn_credit.click(
            fn=predict_credit_score,
            inputs=[edad_credit, genero_credit, ingresos_mensuales_pesos_credit,
                    educacion_credit, estado_civil_credit, num_hijos_credit, propiedad_vivienda_credit],
            outputs=output_credit,
        )

    with gr.Tab("Perfil de Riesgo de Inversión"):
        gr.Markdown("## Reconozca Su Perfil de Riesgo")
        gr.Markdown("""
        Para nadie es un secreto que en el mercado de capitales se puede ganar mucho dinero pero también es
        verdad que existe un riesgo, igual de grande, de perder toda la inversión. Cada activo o título que se negocia
        en el mercado posee un riesgo que depende de distintos factores, en general, los títulos de renta variable son
        más volátiles y, por ende, más riesgosos que otros títulos como los de renta fija.

        Son diferentes los activos que se pueden tomar como vehículos de inversión, y para usted entrar a invertir en
        alguna (o todas) las posibilidades debe crear primero una cuenta en comisionista de bolsa, una fiduciaria o un
        banco de inversión. Sin embargo, debe tener en cuenta el riesgo de cada uno de los activos ya que de esto
        dependerá la probabilidad de que usted gane o pierda dinero. Así, después de haber hecho todos los trámites
        burocráticos y el depósito del dinero en su cuenta de inversión, el siguiente paso antes de invertir será
        conocer su perfil de riesgo.

        **¿Qué es el perfil de riesgo?**

        El perfil de riesgo de una persona indica la capacidad de asumir pérdidas dependiendo de la rentabilidad que
        pueda obtener de una inversión. Es decir, dependiendo de qué tanto riesgo esté dispuesto a asumir con
        respecto a sus inversiones podrá tener desde un perfil de riesgo moderado a uno agresivo.

        **Perfiles de riesgo**

        - **Perfil Conservador:** Prefiere inversiones seguras con baja rentabilidad.
        - **Perfil Moderado:** Equilibrio entre riesgo y rentabilidad.
        - **Perfil Agresivo:** Alto riesgo con potencial de alta rentabilidad.

        **¿Cómo saber su perfil de riesgo?**

        Es importante tener una idea de qué tan arriesgado es en realidad. Para ello, lo invito a realizar la siguiente
        encuesta con el fin de conocer la tolerancia que tiene al riesgo:
        """)

        # Preguntas del cuestionario
        pregunta_1_inv = gr.Radio(
            choices=[
                "a. Vender y evitar una mayor pérdida, probar con otro activo.",
                "b. No hacer nada y esperar a que la inversión se recupere.",
                "c. Comprar más. Fue una buena inversión antes y ahora es buena hora de comprar más."
            ],
            label="1. A 60 días después de depositar dinero en una inversión, su precio cae un 20%. Suponiendo que la información relevante no ha cambiado, ¿qué haría usted?"
        )

        pregunta_2_inv = gr.Radio(
            choices=[
                "a. Vender.",
                "b. No hacer nada.",
                "c. Comprar más."
            ],
            label="2. Su inversión se redujo un 20%, pero es parte de una cartera que se utiliza para cumplir con metas de inversión de tres diferentes horizontes temporales. ¿Qué haría usted si el objetivo era de cinco años?"
        )

        pregunta_3_inv = gr.Radio(
            choices=[
                "a. Venderlos y materializar sus ganancias.",
                "b. No hacer nada, esperar que aumente más.",
                "c. Comprar más, podría ir más alto."
            ],
            label="3. El precio de su inversión de retiro sube un 25% un mes después de haberla comprado. Si la información relevante no ha cambiado y tiene la posibilidad de hacer algo ¿qué haría?"
        )

        pregunta_4_inv = gr.Radio(
            choices=[
                "a. Invertir en un fondo del mercado monetario o fondos garantizados de inversión, renunciando a la posibilidad de mayores ganancias, pero prácticamente asegurando el capital.",
                "b. Invertir en una mezcla de 50-50 de los fondos de bonos y acciones, esperando conseguir algún crecimiento del capital, pero protegiendo de alguna forma un ingreso fijo.",
                "c. Invertir en fondos mutuos agresivos cuyo valor es probable que fluctúen de forma significativa durante el año, pero tienen el potencial para los aumentos impresionantes de cinco a diez años."
            ],
            label="4. Usted está invirtiendo para su jubilación, que es a 15 años. ¿Qué prefiere hacer?"
        )

        pregunta_5_inv = gr.Radio(
            choices=[
                "a. $2000 en efectivo.",
                "b. Uno que gane $5000 (probabilidad del 50%).",
                "c. Uno que gane $15000 (probabilidad del 20%)."
            ],
            label="5. Acaba de ganar un gran premio en un concurso. ¿Cuál escogería?"
        )

        pregunta_6_inv = gr.Radio(
            choices=[
                "a. Definitivamente no.",
                "b. Tal vez.",
                "c. Sí."
            ],
            label="6. Una buena oportunidad de inversión acaba de llegar. Pero usted tiene que pedir prestado el dinero para entrar, ¿sacaría usted un préstamo?"
        )

        pregunta_7_inv = gr.Radio(
            choices=[
                "a. Nada.",
                "b. Dos meses de sueldo.",
                "c. Cuatro meses de sueldo."
            ],
            label="7. Su empresa está vendiendo de acciones a los empleados. En tres años, la compañía planea vender al público la empresa. Hasta entonces, usted no será capaz de vender sus acciones y no recibirá dividendos. Sin embargo, su inversión podría multiplicarse por lo menos 10 veces cuando la empresa salga al público. ¿Qué cantidad de dinero invertiría?"
        )

        monto_inversion = gr.Number(label="Monto a invertir ($)", value=1000)

        btn_inversion = gr.Button("Recomendar Inversión")
        output_inversion = gr.Textbox(label="Recomendación", lines=15)
        output_grafico = gr.Image(label="Distribución Recomendada")

        btn_inversion.click(
            fn=recomendar_inversion,
            inputs=[pregunta_1_inv, pregunta_2_inv, pregunta_3_inv, pregunta_4_inv,
                    pregunta_5_inv, pregunta_6_inv, pregunta_7_inv, monto_inversion],
            outputs=[output_inversion, output_grafico],
        )

    demo.launch()

import math
import random
import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import linregress, norm


# DATOS DEL GRUPO
INTEGRANTES = "CRISTIAN LEMUS"
ASIGNATURA = "FISICA II"
SECCION = "SECCION 1"


st.set_page_config(
    page_title="Estadística Fácil",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Estadística Fácil")
st.write("Calcula, comprende y practica paso a paso.")
st.caption(
    f"Integrantes: {INTEGRANTES} | "
    f"Asignatura: {ASIGNATURA} | Sección: {SECCION}"
)


def leer_datos(texto):
    """Convierte números separados por comas en una lista."""
    partes = texto.split(",")

    if any(not parte.strip() for parte in partes):
        raise ValueError("Faltan datos o hay comas sin un número.")

    try:
        datos = [float(parte.strip()) for parte in partes]
    except ValueError:
        raise ValueError(
            "Escribe solo números separados por comas. "
            "Usa punto para los decimales."
        )

    if not all(math.isfinite(valor) for valor in datos):
        raise ValueError("Los números deben ser finitos.")

    # Límites para evitar operaciones excesivas y desbordamientos.
    if len(datos) > 1000:
        raise ValueError("Ingresa como máximo 1000 datos.")

    if any(abs(valor) > 1e12 for valor in datos):
        raise ValueError("Usa números con valor absoluto menor o igual a 10¹².")

    return datos


def formato(valor):
    return f"{valor:.6g}"


def mostrar_descriptiva(datos, unidad):
    """Calcula y explica las medidas descriptivas."""
    n = len(datos)
    media = statistics.mean(datos)
    mediana = statistics.median(datos)
    rango = max(datos) - min(datos)
    frecuencias = Counter(datos)
    mayor_frecuencia = max(frecuencias.values())
    modas = [
        valor for valor, frecuencia in frecuencias.items()
        if frecuencia == mayor_frecuencia
    ]

    etiqueta = f" {unidad}" if unidad else ""

    st.subheader("Resultados")
    a, b, c = st.columns(3)
    a.metric("Media", formato(media) + etiqueta)
    b.metric("Mediana", formato(mediana) + etiqueta)
    c.metric("Rango", formato(rango) + etiqueta)

    if mayor_frecuencia == 1:
        st.write("**Moda:** no hay valores que se repitan.")
    else:
        st.write(
            "**Moda(s):** "
            + ", ".join(formato(valor) for valor in sorted(modas))
            + etiqueta
        )

    st.subheader("Procedimiento")

    st.write("**1. Media: sumar los datos y dividir entre su cantidad.**")
    st.latex(r"\bar{x}=\frac{\sum x_i}{n}")
    st.write(
        f"Suma = {formato(sum(datos))}; cantidad = {n}. "
        f"Media = {formato(sum(datos))} / {n} = {formato(media)}"
        f"{etiqueta}."
    )

    ordenados = sorted(datos)
    st.write("**2. Mediana: ordenar los datos y localizar el centro.**")
    st.write("Datos ordenados:", ordenados)

    if n % 2:
        st.write(
            f"Posición central: {n // 2 + 1}. "
            f"Mediana = {formato(mediana)}{etiqueta}."
        )
    else:
        izquierda = ordenados[n // 2 - 1]
        derecha = ordenados[n // 2]
        st.write(
            f"Mediana = ({formato(izquierda)} + "
            f"{formato(derecha)}) / 2 = "
            f"{formato(mediana)}{etiqueta}."
        )

    st.write("**3. Moda: encontrar los valores con mayor frecuencia.**")
    st.dataframe(
        pd.DataFrame(
            sorted(frecuencias.items()),
            columns=["Valor", "Frecuencia"],
        ),
        hide_index=True,
    )

    st.write("**4. Rango: máximo menos mínimo.**")
    st.write(
        f"{formato(max(datos))} − {formato(min(datos))} "
        f"= {formato(rango)}{etiqueta}."
    )

    st.write("**5. Dispersión: calcular las diferencias respecto a la media.**")
    diferencias = [valor - media for valor in datos]
    cuadrados = [valor ** 2 for valor in diferencias]
    suma_cuadrados = sum(cuadrados)

    st.dataframe(
        pd.DataFrame({
            "Dato": datos,
            "Dato − media": diferencias,
            "(Dato − media)²": cuadrados,
        }),
        hide_index=True,
    )

    st.write(
        f"Suma de diferencias al cuadrado = "
        f"{formato(suma_cuadrados)}."
    )

    var_p = statistics.pvariance(datos)
    desv_p = statistics.pstdev(datos)
    unidad_cuadrada = f" ({unidad})²" if unidad else ""

    st.latex(
        r"\sigma^2=\frac{\sum(x_i-\bar{x})^2}{n}"
        r"\qquad \sigma=\sqrt{\sigma^2}"
    )
    st.write(
        f"Varianza poblacional = {formato(suma_cuadrados)} / {n} "
        f"= {formato(var_p)}{unidad_cuadrada}."
    )
    st.write(
        f"Desviación poblacional = √{formato(var_p)} "
        f"= {formato(desv_p)}{etiqueta}."
    )

    if n >= 2:
        var_m = statistics.variance(datos)
        desv_m = statistics.stdev(datos)

        st.latex(
            r"s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}"
            r"\qquad s=\sqrt{s^2}"
        )
        st.write(
            f"Varianza muestral = {formato(suma_cuadrados)} / {n - 1} "
            f"= {formato(var_m)}{unidad_cuadrada}."
        )
        st.write(
            f"Desviación muestral = √{formato(var_m)} "
            f"= {formato(desv_m)}{etiqueta}."
        )
        st.info(
            f"Los datos tienen una media de {formato(media)}{etiqueta}. "
            f"La desviación muestral de {formato(desv_m)}{etiqueta} "
            "describe su dispersión alrededor de esa media."
        )
    else:
        st.warning(
            "Se necesitan al menos dos datos para calcular "
            "varianza y desviación estándar muestrales."
        )

    st.write(
        "**¿Cuál utilizar?** La poblacional divide entre n y se utiliza "
        "si los datos representan toda la población estudiada. "
        "La muestral divide entre n − 1 y se utiliza para estimar "
        "la dispersión de una población a partir de una muestra."
    )


menu = st.sidebar.radio(
    "Selecciona una función",
    [
        "Estadística y mediciones",
        "Tablas y gráficas",
        "Probabilidad",
        "Correlación y regresión",
        "Ejemplos",
        "Practicar",
    ],
)


if menu == "Estadística y mediciones":
    st.header("Estadística y mediciones")
    st.write(
        "Introduce los datos en una misma unidad. "
        "Si tienes unidades distintas, conviértelas antes."
    )

    texto = st.text_area(
        "Datos separados por comas",
        "2.1, 2.3, 2.2, 2.4, 2.0",
    )
    unidad = st.text_input("Unidad, por ejemplo A, s, m o kg", "A")
    st.caption("Usa punto para los decimales.")

    if st.button("Calcular estadísticas"):
        try:
            mostrar_descriptiva(leer_datos(texto), unidad.strip())
        except ValueError as error:
            st.error(str(error))


elif menu == "Tablas y gráficas":
    st.header("Tablas y gráficas")
    texto = st.text_area(
        "Datos separados por comas",
        "1, 2, 2, 3, 3, 3, 4, 4, 5",
    )
    unidad = st.text_input("Unidad o descripción del eje", "Valor")
    intervalos = st.slider("Cantidad de intervalos del histograma", 1, 20, 5)

    if st.button("Construir tabla e histograma"):
        try:
            datos = leer_datos(texto)
            conteo = Counter(datos)
            valores = sorted(conteo)
            frecuencias = [conteo[valor] for valor in valores]

            tabla = pd.DataFrame({
                "Valor": valores,
                "Frecuencia": frecuencias,
                "Frecuencia relativa": [
                    frecuencia / len(datos) for frecuencia in frecuencias
                ],
                "Frecuencia acumulada": np.cumsum(frecuencias),
            })

            st.dataframe(tabla, hide_index=True)
            st.write(
                "**Procedimiento:** contamos las apariciones de cada valor. "
                "La frecuencia relativa es frecuencia / total de datos. "
                "La acumulada suma las frecuencias hasta cada fila."
            )
            st.write(
                f"Total de datos: {len(datos)}. "
                "Las frecuencias relativas suman 1."
            )

            figura, eje = plt.subplots()
            eje.hist(datos, bins=intervalos, edgecolor="black")
            eje.set_xlabel(unidad)
            eje.set_ylabel("Frecuencia")
            eje.set_title("Histograma")
            st.pyplot(figura)
            plt.close(figura)

            st.caption(
                "El histograma agrupa los datos en intervalos. "
                "La tabla anterior muestra frecuencias por valor."
            )
        except ValueError as error:
            st.error(str(error))

    st.info(
        "El diagrama de dispersión se encuentra en "
        "«Correlación y regresión», donde puedes ingresar X e Y."
    )


elif menu == "Probabilidad":
    st.header("Probabilidad")
    tipo = st.selectbox(
        "Tipo de ejercicio",
        ["Probabilidad básica", "Distribución normal"],
    )

    if tipo == "Probabilidad básica":
        st.caption("Este modelo supone resultados igualmente probables.")
        favorables = st.number_input(
            "Casos favorables", min_value=0, value=3, step=1
        )
        posibles = st.number_input(
            "Casos posibles", min_value=1, value=6, step=1
        )

        if st.button("Calcular probabilidad"):
            if favorables > posibles:
                st.error("Los casos favorables no pueden superar los posibles.")
            else:
                probabilidad = favorables / posibles
                st.latex(r"P(A)=\frac{\text{favorables}}{\text{posibles}}")
                st.write(
                    f"P(A) = {favorables} / {posibles} "
                    f"= {probabilidad:.6f}"
                )
                st.success(f"Probabilidad: {probabilidad:.2%}")
                st.write(
                    f"Interpretación: el evento representa "
                    f"{probabilidad * 100:.2f}% de los resultados posibles."
                )

    else:
        media = st.number_input("Media μ", value=100.0)
        desviacion = st.number_input("Desviación estándar σ", value=15.0)
        operacion = st.selectbox(
            "Probabilidad que deseas calcular",
            ["P(X ≤ a)", "P(X > a)", "P(a ≤ X ≤ b)"],
        )
        limite_a = st.number_input("Valor a", value=100.0)
        limite_b = None

        if operacion == "P(a ≤ X ≤ b)":
            limite_b = st.number_input("Valor b", value=115.0)

        if st.button("Resolver distribución normal"):
            if desviacion <= 0:
                st.error("La desviación estándar debe ser mayor que cero.")
            elif limite_b is not None and limite_b < limite_a:
                st.error("El límite b debe ser mayor o igual que a.")
            else:
                za = (limite_a - media) / desviacion
                st.latex(r"z=\frac{x-\mu}{\sigma}")
                st.write(
                    f"z(a) = ({limite_a} − {media}) / "
                    f"{desviacion} = {za:.6f}"
                )

                if operacion == "P(X ≤ a)":
                    probabilidad = norm.cdf(za)
                    st.write(f"P = Φ({za:.6f})")
                elif operacion == "P(X > a)":
                    probabilidad = norm.sf(za)
                    st.write(f"P = 1 − Φ({za:.6f})")
                else:
                    zb = (limite_b - media) / desviacion
                    st.write(
                        f"z(b) = ({limite_b} − {media}) / "
                        f"{desviacion} = {zb:.6f}"
                    )
                    # Evita restar números cercanos a 1 en la cola derecha.
                    if za >= 0:
                        probabilidad = norm.sf(za) - norm.sf(zb)
                    else:
                        probabilidad = norm.cdf(zb) - norm.cdf(za)
                    st.write(f"P = Φ({zb:.6f}) − Φ({za:.6f})")

                st.caption("Φ es la función acumulada de la normal estándar.")
                st.success(f"Probabilidad: {probabilidad:.6%}")
                st.write(
                    "El resultado representa el porcentaje de valores "
                    "que cumple la condición bajo el modelo normal indicado."
                )


elif menu == "Correlación y regresión":
    st.header("Correlación y regresión")
    texto_x = st.text_input("Datos de X", "1, 2, 3, 4, 5")
    texto_y = st.text_input("Datos de Y", "2, 4, 5, 4, 5")
    st.caption("Cada valor de X debe tener su correspondiente valor de Y.")

    if st.button("Analizar relación"):
        try:
            x = leer_datos(texto_x)
            y = leer_datos(texto_y)

            if len(x) != len(y):
                raise ValueError("X e Y deben tener la misma cantidad de datos.")
            if len(x) < 3:
                raise ValueError("Ingresa al menos tres pares de datos.")
            if len(set(x)) < 2:
                raise ValueError("X debe contener valores diferentes.")
            if len(set(y)) < 2:
                raise ValueError(
                    "Y es constante: la correlación no está definida. "
                    "Utiliza datos con variación para este análisis."
                )

            resultado = linregress(x, y)
            promedio_x = statistics.mean(x)
            promedio_y = statistics.mean(y)
            dx = [valor - promedio_x for valor in x]
            dy = [valor - promedio_y for valor in y]
            sxx = sum(valor ** 2 for valor in dx)
            syy = sum(valor ** 2 for valor in dy)
            sxy = sum(a * b for a, b in zip(dx, dy))

            st.dataframe(
                pd.DataFrame({
                    "X": x,
                    "Y": y,
                    "X − media X": dx,
                    "Y − media Y": dy,
                    "Producto": [a * b for a, b in zip(dx, dy)],
                }),
                hide_index=True,
            )

            st.write(
                f"Media X = {formato(promedio_x)}; "
                f"media Y = {formato(promedio_y)}."
            )
            st.write(
                f"Sxx = {formato(sxx)}; Syy = {formato(syy)}; "
                f"Sxy = {formato(sxy)}."
            )
            st.latex(
                r"b=\frac{S_{xy}}{S_{xx}},\quad "
                r"a=\bar{y}-b\bar{x},\quad "
                r"r=\frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}"
            )
            st.write(
                f"b = {formato(sxy)} / {formato(sxx)} "
                f"= {formato(resultado.slope)}"
            )
            st.write(
                f"a = {formato(promedio_y)} − "
                f"({formato(resultado.slope)} × {formato(promedio_x)}) "
                f"= {formato(resultado.intercept)}"
            )
            st.write(
                f"r = {formato(sxy)} / "
                f"√({formato(sxx)} × {formato(syy)}) "
                f"= {formato(resultado.rvalue)}"
            )
            st.success(
                f"Recta: ŷ = {resultado.intercept:.4f} "
                f"+ ({resultado.slope:.4f}) × X"
            )
            st.write(f"**R² = {resultado.rvalue ** 2:.4f}**")

            if abs(resultado.rvalue) < 1e-12:
                st.info("En estos datos no se observa correlación lineal.")
            else:
                sentido = "positiva" if resultado.rvalue > 0 else "negativa"
                st.info(
                    f"La correlación es {sentido}. Cuanto más cercano "
                    "esté |r| a 1, mayor es la asociación lineal; "
                    "cuanto más cercano a 0, menor es esa asociación."
                )

            st.write(
                f"La pendiente indica un cambio estimado de "
                f"{resultado.slope:.4f} unidades de Y por cada unidad "
                "adicional de X. Correlación no implica causalidad."
            )

            figura, eje = plt.subplots()
            eje.scatter(x, y, label="Datos")
            linea_x = np.linspace(min(x), max(x), 100)
            linea_y = resultado.intercept + resultado.slope * linea_x
            eje.plot(linea_x, linea_y, color="red", label="Regresión")
            eje.set_xlabel("X")
            eje.set_ylabel("Y")
            eje.legend()
            st.pyplot(figura)
            plt.close(figura)

        except ValueError as error:
            st.error(str(error))


elif menu == "Ejemplos":
    st.header("10 ejercicios de ejemplo")
    st.write(
        "Copia los datos en el módulo indicado para obtener "
        "el procedimiento completo."
    )

    ejemplos = [
        ("Media", "Datos: 2, 4, 6, 8. Media = 5."),
        ("Mediana", "Datos: 9, 1, 5. Ordenados: 1, 5, 9. Mediana = 5."),
        ("Moda", "Datos: 2, 3, 3, 5. Moda = 3."),
        ("Rango", "Datos: 4, 7, 10. Rango = 10 − 4 = 6."),
        (
            "Dispersión poblacional",
            "Datos: 2, 4, 6. Varianza poblacional = 8/3 ≈ 2.666667; "
            "desviación poblacional ≈ 1.632993.",
        ),
        (
            "Mediciones de corriente",
            "Datos: 2.1, 2.3, 2.2, 2.4, 2.0. Unidad: A. "
            "Media = 2.2 A; desviación muestral ≈ 0.158114 A.",
        ),
        (
            "Frecuencias",
            "Datos: 1, 1, 2, 3, 3, 3. "
            "Frecuencias: 1 → 2; 2 → 1; 3 → 3.",
        ),
        (
            "Probabilidad básica",
            "Dado equilibrado: obtener un número par. "
            "Favorables = 3; posibles = 6. P = 0.5 = 50%.",
        ),
        (
            "Distribución normal",
            "Media = 100; desviación = 15; "
            "P(X ≤ 115): z = 1; probabilidad ≈ 84.1345%.",
        ),
        (
            "Correlación y regresión",
            "X: 1, 2, 3, 4. Y: 2, 4, 6, 8. "
            "r = 1; recta: ŷ = 2X; R² = 1.",
        ),
    ]

    for numero, (titulo, descripcion) in enumerate(ejemplos, start=1):
        with st.expander(f"{numero}. {titulo}"):
            st.write(descripcion)


elif menu == "Practicar":
    st.header("Prácticas con retroalimentación")
    tema = st.selectbox(
        "¿Qué deseas practicar?",
        ["Media", "Mediana", "Rango", "Desviación muestral", "Probabilidad"],
    )

    if st.button("Generar ejercicio"):
        datos = [random.randint(1, 20) for _ in range(5)]

        if tema == "Probabilidad":
            posibles = random.randint(4, 20)
            favorables = random.randint(0, posibles)
            pregunta = (
                f"Hay {favorables} casos favorables de {posibles} "
                "resultados igualmente probables. "
                "Calcula la probabilidad en decimal, no en porcentaje."
            )
            correcta = favorables / posibles
            explicacion = (
                f"P = favorables / posibles = "
                f"{favorables} / {posibles} = {correcta:.6f}"
            )
        else:
            pregunta = f"Calcula {tema.lower()} de estos datos: {datos}"

            if tema == "Media":
                correcta = statistics.mean(datos)
                explicacion = (
                    f"Media = {sum(datos)} / {len(datos)} = {correcta:.6f}"
                )
            elif tema == "Mediana":
                correcta = statistics.median(datos)
                explicacion = (
                    f"Ordenados: {sorted(datos)}. "
                    f"El valor central es {correcta:.6f}."
                )
            elif tema == "Rango":
                correcta = max(datos) - min(datos)
                explicacion = (
                    f"Rango = {max(datos)} − {min(datos)} = {correcta:.6f}"
                )
            else:
                correcta = statistics.stdev(datos)
                media = statistics.mean(datos)
                suma = sum((valor - media) ** 2 for valor in datos)
                explicacion = (
                    f"Media = {media:.6f}. Suma de diferencias al cuadrado "
                    f"= {suma:.6f}. s = √({suma:.6f}/4) "
                    f"= {correcta:.6f}"
                )

        st.session_state["ejercicio"] = {
            "pregunta": pregunta,
            "correcta": correcta,
            "explicacion": explicacion,
        }

    if "ejercicio" in st.session_state:
        ejercicio = st.session_state["ejercicio"]
        st.write(ejercicio["pregunta"])
        respuesta = st.text_input("Tu respuesta")
        st.caption("Usa punto decimal. Se acepta un error absoluto de 0.01.")

        if st.button("Revisar respuesta"):
            try:
                numero = float(respuesta)
                if not math.isfinite(numero):
                    raise ValueError

                if abs(numero - ejercicio["correcta"]) <= 0.01:
                    st.success("¡Correcto!")
                else:
                    st.error("Revisa tu respuesta y compara el procedimiento.")

                st.write(ejercicio["explicacion"])
            except ValueError:
                st.error("Ingresa una respuesta numérica válida.")
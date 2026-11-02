import streamlit as st
import pulp
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="DataNet Analytics - Optimizador",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 DataNet Analytics")
st.subheader("Optimización de una red de comunicaciones multiflujo")
st.write(
    "Edita los enlaces y las demandas, ejecuta el modelo de PuLP "
    "y compara diferentes escenarios de la red."
)


# ============================================================
# 2. DATOS INICIALES
# ============================================================

def crear_datos_iniciales():
    # Cada enlace tiene: nodo inicial, nodo final, costo, capacidad y estado.
    datos_enlaces = [
        [1, 2, 4, 12, True],
        [1, 3, 6, 10, True],
        [2, 3, 2, 8, True],
        [2, 4, 5, 10, True],
        [2, 5, 7, 8, True],
        [3, 4, 3, 12, True],
        [3, 6, 5, 10, True],
        [4, 5, 2, 10, True],
        [4, 7, 6, 12, True],
        [5, 6, 3, 8, True],
        [5, 8, 5, 10, True],
        [6, 7, 2, 10, True],
        [6, 8, 4, 12, True],
        [7, 8, 3, 14, True]
    ]

    columnas_enlaces = [
        "Origen", "Destino", "Costo", "Capacidad", "Activo"
    ]

    tabla_enlaces = pd.DataFrame(
        datos_enlaces,
        columns=columnas_enlaces
    )

    # Cada flujo tiene: nombre, origen, destino y demanda.
    datos_flujos = [
        ["F1", 1, 8, 6],
        ["F2", 2, 7, 5],
        ["F3", 3, 8, 4]
    ]

    columnas_flujos = [
        "Flujo", "Origen", "Destino", "Demanda"
    ]

    tabla_flujos = pd.DataFrame(
        datos_flujos,
        columns=columnas_flujos
    )

    return tabla_enlaces, tabla_flujos


if "tabla_enlaces" not in st.session_state:
    st.session_state.tabla_enlaces, st.session_state.tabla_flujos = (
        crear_datos_iniciales()
    )

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "escenarios" not in st.session_state:
    st.session_state.escenarios = []


# ============================================================
# 3. FUNCIONES DEL MODELO
# ============================================================

def resolver_modelo(tabla_enlaces, tabla_flujos):
    """
    Construye y resuelve el modelo de flujo multicommodity.

    Devuelve el estado, el costo mínimo, los flujos por enlace
    y la utilización de las capacidades.
    """

    # Conservar solamente los enlaces que están activos.
    enlaces_activos = []

    costos = {}
    capacidades = {}

    for _, fila in tabla_enlaces.iterrows():
        i = int(fila["Origen"])
        j = int(fila["Destino"])

        if bool(fila["Activo"]):
            enlaces_activos.append((i, j))
            costos[(i, j)] = float(fila["Costo"])
            capacidades[(i, j)] = float(fila["Capacidad"])

    # Leer los datos de cada comunicación.
    flujos = {}

    for _, fila in tabla_flujos.iterrows():
        nombre = str(fila["Flujo"])
        origen = int(fila["Origen"])
        destino = int(fila["Destino"])
        demanda = float(fila["Demanda"])

        flujos[nombre] = (origen, destino, demanda)

    # Verificar que los datos sean válidos.
    if len(enlaces_activos) == 0:
        return {"estado": "Error: no hay enlaces activos."}

    for (i, j) in enlaces_activos:
        if capacidades[(i, j)] <= 0:
            return {"estado": f"Error: capacidad no válida en el enlace {i} → {j}."}
        if costos[(i, j)] < 0:
            return {"estado": f"Error: costo negativo en el enlace {i} → {j}."}

    for nombre, (origen, destino, demanda) in flujos.items():
        if origen == destino:
            return {"estado": f"Error: el origen y destino de {nombre} son iguales."}
        if demanda < 0:
            return {"estado": f"Error: demanda negativa en {nombre}."}

    # Crear el modelo de minimización.
    modelo = pulp.LpProblem(
        "DataNet_Analytics",
        pulp.LpMinimize
    )

    # --------------------------------------------------------
    # Variables de decisión
    # X[flujo, i, j] = Gbps del flujo que pasan por el enlace.
    # --------------------------------------------------------

    X = {}

    for nombre_flujo in flujos:
        for (i, j) in enlaces_activos:
            X[(nombre_flujo, i, j)] = pulp.LpVariable(
                f"X_{nombre_flujo}_{i}_{j}",
                lowBound=0,
                cat="Continuous"
            )

    # --------------------------------------------------------
    # Función objetivo: minimizar el costo total.
    # --------------------------------------------------------

    modelo += pulp.lpSum(
        costos[(i, j)] * X[(nombre_flujo, i, j)]
        for nombre_flujo in flujos
        for (i, j) in enlaces_activos
    ), "Costo_Total"

    # --------------------------------------------------------
    # Restricciones de capacidad compartida.
    # --------------------------------------------------------

    for (i, j) in enlaces_activos:
        modelo += (
            pulp.lpSum(
                X[(nombre_flujo, i, j)]
                for nombre_flujo in flujos
            )
            <= capacidades[(i, j)]
        ), f"Capacidad_{i}_{j}"

    # --------------------------------------------------------
    # Conservación de flujo para cada flujo y cada nodo.
    # Saliente - Entrante = balance.
    # --------------------------------------------------------

    nodos = [1, 2, 3, 4, 5, 6, 7, 8]

    for nombre_flujo in flujos:

        origen, destino, demanda = flujos[nombre_flujo]

        for nodo in nodos:

            flujo_saliente = pulp.lpSum(
                X[(nombre_flujo, i, j)]
                for (i, j) in enlaces_activos
                if i == nodo
            )

            flujo_entrante = pulp.lpSum(
                X[(nombre_flujo, i, j)]
                for (i, j) in enlaces_activos
                if j == nodo
            )

            if nodo == origen:
                balance = demanda
            elif nodo == destino:
                balance = -demanda
            else:
                balance = 0

            modelo += (
                flujo_saliente - flujo_entrante == balance
            ), f"Conservacion_{nombre_flujo}_{nodo}"

    # Resolver con CBC.
    solucionador = pulp.PULP_CBC_CMD(msg=False)

    try:
        modelo.solve(solucionador)
    except Exception as error:
        return {
            "estado": f"Error al ejecutar el solucionador: {error}"
        }

    estado = pulp.LpStatus[modelo.status]

    if estado != "Optimal":
        return {
            "estado": estado,
            "mensaje": (
                "No se encontró una solución óptima. "
                "Comprueba las capacidades, demandas y enlaces activos."
            )
        }

    # --------------------------------------------------------
    # Preparar los resultados.
    # --------------------------------------------------------

    costo_total = float(pulp.value(modelo.objective))

    resultados_flujos = []

    for nombre_flujo in flujos:
        for (i, j) in enlaces_activos:

            cantidad = pulp.value(X[(nombre_flujo, i, j)])

            if cantidad is not None and cantidad > 0.000001:
                resultados_flujos.append({
                    "Flujo": nombre_flujo,
                    "Origen enlace": i,
                    "Destino enlace": j,
                    "Tráfico (Gbps)": cantidad
                })

    resultados_enlaces = []

    for _, fila in tabla_enlaces.iterrows():

        i = int(fila["Origen"])
        j = int(fila["Destino"])
        activo = bool(fila["Activo"])

        if activo:
            trafico_total = sum(
                pulp.value(X[(nombre_flujo, i, j)]) or 0
                for nombre_flujo in flujos
            )
            capacidad = float(fila["Capacidad"])
            utilizacion = (
                trafico_total / capacidad * 100
                if capacidad > 0 else 0
            )
        else:
            trafico_total = 0
            capacidad = float(fila["Capacidad"])
            utilizacion = 0

        resultados_enlaces.append({
            "Origen": i,
            "Destino": j,
            "Activo": activo,
            "Costo": float(fila["Costo"]),
            "Capacidad": capacidad,
            "Tráfico total (Gbps)": trafico_total,
            "Utilización (%)": utilizacion
        })

    return {
        "estado": estado,
        "costo_total": costo_total,
        "flujos": pd.DataFrame(resultados_flujos),
        "enlaces": pd.DataFrame(resultados_enlaces)
    }


# ============================================================
# 4. VISUALIZACIÓN DE LA RED
# ============================================================

def dibujar_red(tabla_enlaces, resultado=None):
    # Posiciones fijas para organizar los ocho nodos.
    posiciones = {
        1: (0, 2),
        2: (1.5, 3),
        3: (1.5, 1),
        4: (3, 3),
        5: (4.5, 3),
        6: (4.5, 1),
        7: (6, 2),
        8: (7.5, 2)
    }

    datos_resultados = {}

    if resultado is not None and "enlaces" in resultado:
        for _, fila in resultado["enlaces"].iterrows():
            datos_resultados[
                (int(fila["Origen"]), int(fila["Destino"]))
            ] = fila

    figura = go.Figure()

    # Dibujar cada enlace con un color según su estado.
    for _, fila in tabla_enlaces.iterrows():

        i = int(fila["Origen"])
        j = int(fila["Destino"])

        x1, y1 = posiciones[i]
        x2, y2 = posiciones[j]

        activo = bool(fila["Activo"])
        color = "#9CA3AF"
        grosor = 2
        estilo = "dash"
        etiqueta = "Desactivado"

        if activo:
            color = "#9CA3AF"
            estilo = "solid"
            etiqueta = "Disponible"

        if (i, j) in datos_resultados and activo:

            dato = datos_resultados[(i, j)]
            trafico = float(dato["Tráfico total (Gbps)"])
            capacidad = float(dato["Capacidad"])
            utilizacion = float(dato["Utilización (%)"])

            if utilizacion >= 99.999:
                color = "#DC2626"
                grosor = 5
                etiqueta = "Saturado"
            elif trafico > 0.000001:
                color = "#2563EB"
                grosor = 4
                etiqueta = "Utilizado"

            texto = (
                f"{i} → {j}<br>"
                f"Costo: {fila['Costo']}<br>"
                f"Tráfico: {trafico:.2f} / {capacidad:.2f} Gbps<br>"
                f"Utilización: {utilizacion:.2f}%<br>"
                f"Estado: {etiqueta}"
            )
        else:
            texto = (
                f"{i} → {j}<br>"
                f"Costo: {fila['Costo']}<br>"
                f"Capacidad: {fila['Capacidad']} Gbps<br>"
                f"Estado: {etiqueta}"
            )

        figura.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(color=color, width=grosor, dash=estilo),
            hoverinfo="text",
            text=texto,
            showlegend=False
        ))

        # Flecha para indicar la dirección del enlace.
        figura.add_annotation(
            x=x2,
            y=y2,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=color,
            standoff=15,
            startstandoff=15
        )

        # Etiqueta del enlace.
        punto_medio_x = (x1 + x2) / 2
        punto_medio_y = (y1 + y2) / 2

        figura.add_annotation(
            x=punto_medio_x,
            y=punto_medio_y + 0.13,
            text=f"{i}-{j}",
            showarrow=False,
            font=dict(size=10, color="#374151"),
            bgcolor="white"
        )

    # Dibujar los nodos.
    for nodo, (x, y) in posiciones.items():

        figura.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(
                size=34,
                color="#0F766E",
                line=dict(color="white", width=2)
            ),
            text=[str(nodo)],
            textposition="middle center",
            textfont=dict(color="white", size=13),
            hovertemplate=f"Servidor {nodo}<extra></extra>",
            showlegend=False
        ))

    figura.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.5, 8]),
        yaxis=dict(visible=False, range=[0.3, 3.8]),
        showlegend=False
    )

    return figura


# ============================================================
# 5. EDICIÓN DE DATOS
# ============================================================

st.header("1. Configuración de la red")

columna_enlaces, columna_flujos = st.columns([2, 1])

with columna_enlaces:
    st.subheader("Enlaces de la red")
    st.caption(
        "Modifica los costos, las capacidades y activa o desactiva enlaces."
    )

    enlaces_editados = st.data_editor(
        st.session_state.tabla_enlaces,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="editor_enlaces",
        column_config={
            "Origen": st.column_config.NumberColumn(
                disabled=True
            ),
            "Destino": st.column_config.NumberColumn(
                disabled=True
            ),
            "Costo": st.column_config.NumberColumn(
                min_value=0,
                step=1
            ),
            "Capacidad": st.column_config.NumberColumn(
                min_value=0.01,
                step=1
            ),
            "Activo": st.column_config.CheckboxColumn()
        }
    )

with columna_flujos:
    st.subheader("Demandas de comunicación")
    st.caption("Modifica la demanda de cada flujo.")

    flujos_editados = st.data_editor(
        st.session_state.tabla_flujos,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="editor_flujos",
        column_config={
            "Flujo": st.column_config.TextColumn(
                disabled=True
            ),
            "Origen": st.column_config.NumberColumn(
                disabled=True
            ),
            "Destino": st.column_config.NumberColumn(
                disabled=True
            ),
            "Demanda": st.column_config.NumberColumn(
                min_value=0,
                step=1
            )
        }
    )

# Guardar los datos editados en la sesión.
st.session_state.tabla_enlaces = enlaces_editados.copy()
st.session_state.tabla_flujos = flujos_editados.copy()


# ============================================================
# 6. VISUALIZACIÓN Y OPTIMIZACIÓN
# ============================================================

st.header("2. Visualización de la red")

st.plotly_chart(
    dibujar_red(st.session_state.tabla_enlaces),
    use_container_width=True
)

st.caption(
    "Vista previa: gris continuo = enlace activo; "
    "gris discontinuo = enlace desactivado. "
    "Después de optimizar, se mostrarán los enlaces utilizados y saturados."
)

columna_boton, columna_espacio = st.columns([1, 3])

with columna_boton:
    ejecutar = st.button(
        "▶ Ejecutar optimizador",
        type="primary",
        use_container_width=True
    )

if ejecutar:

    with st.spinner("Resolviendo el modelo de optimización..."):
        st.session_state.resultado = resolver_modelo(
            st.session_state.tabla_enlaces,
            st.session_state.tabla_flujos
        )

    # El resultado corresponde a los datos actuales.
    st.session_state.resultado["datos_enlaces"] = (
        st.session_state.tabla_enlaces.copy()
    )
    st.session_state.resultado["datos_flujos"] = (
        st.session_state.tabla_flujos.copy()
    )


# ============================================================
# 7. PRESENTACIÓN DE RESULTADOS
# ============================================================

resultado = st.session_state.resultado

if resultado is not None:

    st.header("3. Resultados de la optimización")

    if resultado["estado"] == "Optimal":

        costo = resultado["costo_total"]

        enlaces_resultado = resultado["enlaces"]
        flujos_resultado = resultado["flujos"]

        # Indicadores principales.
        enlaces_usados = len(
            enlaces_resultado[
                enlaces_resultado["Tráfico total (Gbps)"] > 0.000001
            ]
        )

        enlaces_saturados = len(
            enlaces_resultado[
                enlaces_resultado["Utilización (%)"] >= 99.999
            ]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Costo total mínimo", f"{costo:,.2f}")
        c2.metric("Enlaces utilizados", enlaces_usados)
        c3.metric("Enlaces saturados", enlaces_saturados)

        # Red con estados calculados.
        st.subheader("Estado de los enlaces")

        st.plotly_chart(
            dibujar_red(
                resultado["datos_enlaces"],
                resultado
            ),
            use_container_width=True
        )

        # Leyenda de colores.
        st.markdown(
            """
            **Leyenda:**
            <span style="color:#9CA3AF">●</span> Disponible sin uso &nbsp;
            <span style="color:#2563EB">●</span> Utilizado &nbsp;
            <span style="color:#DC2626">●</span> Saturado &nbsp;
            <span style="color:#9CA3AF">┄</span> Desactivado
            """,
            unsafe_allow_html=True
        )

        # Distribución de los flujos.
        st.subheader("Distribución de cada comunicación")

        if len(flujos_resultado) > 0:
            st.dataframe(
                flujos_resultado.style.format({
                    "Tráfico (Gbps)": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Las demandas actuales son cero; no hay tráfico que distribuir.")

        # Utilización de cada enlace.
        st.subheader("Utilización de capacidad por enlace")

        tabla_utilizacion = enlaces_resultado.copy()

        st.dataframe(
            tabla_utilizacion.style.format({
                "Costo": "{:.2f}",
                "Capacidad": "{:.2f}",
                "Tráfico total (Gbps)": "{:.2f}",
                "Utilización (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Gráfico de utilización.
        figura_utilizacion = go.Figure()

        colores = []

        for _, fila in tabla_utilizacion.iterrows():
            if not fila["Activo"]:
                colores.append("#9CA3AF")
            elif fila["Utilización (%)"] >= 99.999:
                colores.append("#DC2626")
            elif fila["Tráfico total (Gbps)"] > 0.000001:
                colores.append("#2563EB")
            else:
                colores.append("#9CA3AF")

        etiquetas = [
            f"{int(fila['Origen'])} → {int(fila['Destino'])}"
            for _, fila in tabla_utilizacion.iterrows()
        ]

        figura_utilizacion.add_trace(go.Bar(
            x=etiquetas,
            y=tabla_utilizacion["Utilización (%)"],
            marker_color=colores,
            text=[
                f"{valor:.1f}%"
                for valor in tabla_utilizacion["Utilización (%)"]
            ],
            textposition="outside"
        ))

        figura_utilizacion.update_layout(
            title="Porcentaje de utilización de cada enlace",
            xaxis_title="Enlace",
            yaxis_title="Utilización (%)",
            yaxis=dict(range=[0, 115]),
            height=400
        )

        st.plotly_chart(
            figura_utilizacion,
            use_container_width=True
        )

        # Guardar el escenario actual para compararlo.
        st.subheader("Guardar escenario")

        nombre_escenario = st.text_input(
            "Nombre del escenario",
            value=f"Escenario {len(st.session_state.escenarios) + 1}",
            key="nombre_escenario"
        )

        if st.button("Guardar escenario para comparar"):

            nuevo_escenario = {
                "nombre": nombre_escenario,
                "costo": costo,
                "enlaces": resultado["enlaces"].copy(),
                "flujos": resultado["flujos"].copy()
            }

            st.session_state.escenarios.append(nuevo_escenario)

            st.success(
                f"Escenario '{nombre_escenario}' guardado correctamente."
            )

    else:
        st.error(f"Estado del modelo: {resultado['estado']}")

        if "mensaje" in resultado:
            st.warning(resultado["mensaje"])


# ============================================================
# 8. COMPARACIÓN DE ESCENARIOS
# ============================================================

st.header("4. Comparación de escenarios")

if len(st.session_state.escenarios) == 0:

    st.info(
        "Todavía no hay escenarios guardados. "
        "Ejecuta el optimizador y guarda un escenario para comenzar."
    )

else:

    # Tabla comparativa de costos.
    datos_comparacion = []

    for escenario in st.session_state.escenarios:

        utilizacion_promedio = (
            escenario["enlaces"]["Utilización (%)"].mean()
        )

        cantidad_saturados = len(
            escenario["enlaces"][
                escenario["enlaces"]["Utilización (%)"] >= 99.999
            ]
        )

        datos_comparacion.append({
            "Escenario": escenario["nombre"],
            "Costo total": escenario["costo"],
            "Utilización promedio (%)": utilizacion_promedio,
            "Enlaces saturados": cantidad_saturados
        })

    tabla_comparacion = pd.DataFrame(datos_comparacion)

    st.dataframe(
        tabla_comparacion.style.format({
            "Costo total": "{:.2f}",
            "Utilización promedio (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Gráfico para comparar los costos.
    figura_costos = go.Figure()

    figura_costos.add_trace(go.Bar(
        x=tabla_comparacion["Escenario"],
        y=tabla_comparacion["Costo total"],
        text=[
            f"{valor:.2f}"
            for valor in tabla_comparacion["Costo total"]
        ],
        textposition="outside"
    ))

    figura_costos.update_layout(
        title="Comparación del costo total por escenario",
        xaxis_title="Escenario",
        yaxis_title="Costo total",
        height=350
    )

    st.plotly_chart(
        figura_costos,
        use_container_width=True
    )

    # Permitir seleccionar dos escenarios para comparar
    # la utilización de cada enlace.
    nombres_escenarios = [
        escenario["nombre"]
        for escenario in st.session_state.escenarios
    ]

    if len(nombres_escenarios) >= 2:

        st.subheader("Comparar utilización de dos escenarios")

        col_a, col_b = st.columns(2)

        with col_a:
            escenario_a_nombre = st.selectbox(
                "Primer escenario",
                nombres_escenarios,
                index=0
            )

        with col_b:
            escenario_b_nombre = st.selectbox(
                "Segundo escenario",
                nombres_escenarios,
                index=1
            )

        escenario_a = next(
            escenario for escenario in st.session_state.escenarios
            if escenario["nombre"] == escenario_a_nombre
        )

        escenario_b = next(
            escenario for escenario in st.session_state.escenarios
            if escenario["nombre"] == escenario_b_nombre
        )

        tabla_a = escenario_a["enlaces"]
        tabla_b = escenario_b["enlaces"]

        etiquetas_enlaces = [
            f"{int(fila['Origen'])} → {int(fila['Destino'])}"
            for _, fila in tabla_a.iterrows()
        ]

        figura_comparacion = go.Figure()

        figura_comparacion.add_trace(go.Bar(
            name=escenario_a_nombre,
            x=etiquetas_enlaces,
            y=tabla_a["Utilización (%)"]
        ))

        figura_comparacion.add_trace(go.Bar(
            name=escenario_b_nombre,
            x=etiquetas_enlaces,
            y=tabla_b["Utilización (%)"]
        ))

        figura_comparacion.update_layout(
            title="Utilización de capacidad por enlace",
            xaxis_title="Enlace",
            yaxis_title="Utilización (%)",
            barmode="group",
            height=400
        )

        st.plotly_chart(
            figura_comparacion,
            use_container_width=True
        )

    # Opción para limpiar escenarios guardados.
    if st.button("Eliminar todos los escenarios guardados"):
        st.session_state.escenarios = []
        st.rerun()


# ============================================================
# 9. PIE DE PÁGINA
# ============================================================

st.divider()
st.caption(
    "DataNet Analytics | Modelo de optimización de flujo multicommodity "
    "| Python, Streamlit y PuLP"
)

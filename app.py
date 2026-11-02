import streamlit as st
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide", page_title="DataNet Optimizer")

# 1. INICIALIZACIÓN DE ESTADO (Escenarios)
if "escenarios" not in st.session_state:
    st.session_state.escenarios = {}

# Datos iniciales por defecto
NODOS_DEF = [1, 2, 3, 4, 5, 6, 7, 8]
ENLACES_DEF = [
    (1, 2), (1, 3), (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 6), (4, 5), (4, 7), (5, 6),
    (5, 8), (6, 7), (6, 8), (7, 8)
]
COSTOS_DEF = {
    (1, 2): 4, (1, 3): 6, (2, 3): 2, (2, 4): 5, (2, 5): 7,
    (3, 4): 3, (3, 6): 5, (4, 5): 2, (4, 7): 6, (5, 6): 3,
    (5, 8): 5, (6, 7): 2, (6, 8): 4, (7, 8): 3
}
CAPACIDADES_DEF = {
    (1, 2): 12, (1, 3): 10, (2, 3): 8, (2, 4): 10, (2, 5): 8,
    (3, 4): 12, (3, 6): 10, (4, 5): 10, (4, 7): 12, (5, 6): 8,
    (5, 8): 10, (6, 7): 10, (6, 8): 12, (7, 8): 14
}
FLUJOS_DEF = {
    "F1": (1, 8, 6),
    "F2": (2, 7, 5),
    "F3": (3, 8, 4)
}

st.title("🌐 DataNet Analytics: Optimizador de Redes Multiflujo")

# PANEL LATERAL: Configuración de Datos
st.sidebar.header("⚙️ Configuración de la Red")

# Flujos
st.sidebar.subheader("📌 Demandas de Flujos (Origen, Destino, Gbps)")
flujos = {}
for f_id, (orig_def, dest_def, dem_def) in FLUJOS_DEF.items():
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        orig = col1.number_input(f"{f_id} Origen", min_value=1, max_value=8, value=orig_def, key=f"orig_{f_id}")
    with col2:
        dest = col2.number_input(f"{f_id} Destino", min_value=1, max_value=8, value=dest_def, key=f"dest_{f_id}")
    with col3:
        dem = col3.number_input(f"{f_id} Gbps", min_value=1, max_value=30, value=dem_def, key=f"dem_{f_id}")
    flujos[f_id] = (orig, dest, dem)

# Enlaces activos, costos y capacidades
st.sidebar.subheader("🔗 Gestión de Enlaces")
enlaces_activos = []
costos = {}
capacidades = {}

with st.sidebar.expander("Editar Enlaces (Activar, Costo, Capacidad)", expanded=False):
    for (i, j) in ENLACES_DEF:
        st.markdown(f"**Enlace {i} ➔ {j}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            activo = col1.checkbox("Activo", value=True, key=f"act_{i}_{j}")
        with col2:
            costo = col2.number_input("Costo", min_value=1, value=COSTOS_DEF[(i, j)], key=f"cos_{i}_{j}")
        with col3:
            cap = col3.number_input("Cap.", min_value=1, value=CAPACIDADES_DEF[(i, j)], key=f"cap_{i}_{j}")
        
        if activo:
            enlaces_activos.append((i, j))
            costos[(i, j)] = costo
            capacidades[(i, j)] = cap

# 2. OPTIMIZACIÓN
def optimizar_red(nodos, enlaces, costos, capacidades, flujos):
    modelo = pulp.LpProblem("Optimizacion_DataNet_Analytics", pulp.LpMinimize)
    X = {}
    for flujo in flujos:
        for (i, j) in enlaces:
            X[(flujo, i, j)] = pulp.LpVariable(f"X_{flujo}_{i}_{j}", lowBound=0, cat="Continuous")
            
    # Función Objetivo
    modelo += pulp.lpSum(costos[(i, j)] * X[(flujo, i, j)] for flujo in flujos for (i, j) in enlaces), "Costo_Total"
    
    # Restricciones de Capacidad
    for (i, j) in enlaces:
        modelo += pulp.lpSum(X[(flujo, i, j)] for flujo in flujos) <= capacidades[(i, j)], f"Capacidad_{i}_{j}"
        
    # Restricciones de Conservación de Flujo
    for flujo, (origen, destino, demanda) in flujos.items():
        for nodo in nodos:
            flujo_saliente = pulp.lpSum(X[(flujo, i, j)] for (i, j) in enlaces if i == nodo)
            flujo_entrante = pulp.lpSum(X[(flujo, i, j)] for (i, j) in enlaces if j == nodo)
            if nodo == origen:
                balance = demanda
            elif nodo == destino:
                balance = -demanda
            else:
                balance = 0
            modelo += (flujo_saliente - flujo_entrante == balance), f"Conservacion_{flujo}_Nodo_{nodo}"
            
    modelo.solve(pulp.PULP_CBC_CMD(msg=False))
    return modelo, X

# Ejecutar Optimización
modelo, X = optimizar_red(NODOS_DEF, enlaces_activos, costos, capacidades, flujos)
estado = pulp.LpStatus[modelo.status]

# 3. INTERFAZ PRINCIPAL: DISEÑO EN PESTAÑAS
tab1, tab2 = st.tabs(["📊 Dashboard del Escenario Actual", "📈 Comparativa de Escenarios"])

with tab1:
    col_izq, col_der = st.columns([3, 2])
    
    with col_izq:
        st.subheader("🗺️ Representación Gráfica de la Red")
        
        # Crear Grafo para visualización
        G = nx.DiGraph()
        G.add_nodes_from(NODOS_DEF)
        pos = {1: (0, 1), 2: (1, 2), 3: (1, 0), 4: (2, 1), 5: (3, 2), 6: (3, 0), 7: (4, 1), 8: (5, 1)}
        
        # Listas para clasificación de enlaces en el gráfico
        color_aristas = []
        ancho_aristas = []
        etiquetas_aristas = {}
        
        for (i, j) in ENLACES_DEF:
            if (i, j) not in enlaces_activos:
                color_aristas.append("#D3D3D3") # Desactivado (Gris)
                ancho_aristas.append(1.0)
                etiquetas_aristas[(i, j)] = "Off"
            else:
                trafico = 0.0
                if estado == "Optimal":
                    trafico = sum(pulp.value(X[(f, i, j)]) for f in flujos)
                cap = capacidades[(i, j)]
                util = (trafico / cap) if cap > 0 else 0
                
                etiquetas_aristas[(i, j)] = f"{trafico:.1f}/{cap}"
                
                if util >= 0.99:
                    color_aristas.append("#FF4B4B") # Saturado (Rojo)
                    ancho_aristas.append(4.0)
                elif trafico > 0.001:
                    color_aristas.append("#4CAF50") # En uso (Verde)
                    ancho_aristas.append(2.5)
                else:
                    color_aristas.append("#1F77B4") # Disponible sin uso (Azul estándar)
                    ancho_aristas.append(1.5)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_color="#1F77B4", node_size=700, ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=ENLACES_DEF, edge_color=color_aristas, 
                               width=ancho_aristas, arrows=True, arrowsize=20, ax=ax, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas_aristas, font_size=8, label_pos=0.3, ax=ax)
        
        plt.axis("off")
        st.pyplot(fig)
        
        # Leyenda de Colores
        st.markdown("**Leyenda:** 🔴 Saturado (≥99% Uso) | 🟢 En Uso | 🔵 Disponible | ⚪ Desactivado")

    with col_der:
        st.subheader("📝 Resultados de Optimización")
        if estado == "Optimal":
            costo_total = pulp.value(modelo.objective)
            st.metric(label="Costo Total de Transmisión", value=f"${costo_total:,.2f} USD")
            
            # Guardar Escenario
            st.markdown("---")
            nombre_escenario = st.text_input("Nombre del Escenario Actual", value="Escenario A")
            if st.button("💾 Guardar Escenario Actual"):
                st.session_state.escenarios[nombre_escenario] = {
                    "costo": costo_total,
                    "enlaces": enlaces_activos.copy(),
                    "capacidades": capacidades.copy(),
                    "flujos_X": {k: pulp.value(v) for k, v in X.items()}
                }
                st.success(f"¡{nombre_escenario} guardado exitosamente!")
                
            # Detalles de los Flujos
            st.markdown("### 🔀 Distribución de Flujos")
            for f_id in flujos:
                with st.expander(f"Tráfico detallado de {f_id}", expanded=True):
                    for (i, j) in enlaces_activos:
                        cant = pulp.value(X[(f_id, i, j)])
                        if cant > 0.001:
                            st.write(f"🔹 Enlace {i} ➔ {j}: **{cant:.2f} Gbps**")
        else:
            st.error("⚠️ El modelo no encontró una solución óptima con las restricciones vigentes.")

with tab2:
    st.subheader("📊 Análisis Comparativo de Escenarios")
    if not st.session_state.escenarios:
        st.info("Aún no has guardado ningún escenario. Usa el botón 'Guardar Escenario Actual' en la pestaña anterior para comparar distintos análisis.")
    else:
        datos_comparativa = []
        for nombre, datos in st.session_state.escenarios.items():
            # Calcular métricas globales del escenario
            costo = datos["costo"]
            enlaces_utilizados = 0
            enlaces_saturados = 0
            
            for (i, j) in datos["enlaces"]:
                trafico_total = sum(datos["flujos_X"].get((f, i, j), 0.0) for f in flujos)
                cap = datos["capacidades"].get((i, j), 1)
                if trafico_total > 0.001:
                    enlaces_utilizados += 1
                if (trafico_total / cap) >= 0.99:
                    enlaces_saturados += 1
            
            datos_comparativa.append({
                "Escenario": nombre,
                "Costo Total ($)": round(costo, 2),
                "Enlaces Activos": len(datos["enlaces"]),
                "Enlaces Utilizados": enlaces_utilizados,
                "Enlaces Saturados (100%)": enlaces_saturados
            })
        
        df_comp = pd.DataFrame(datos_comparativa)
        st.dataframe(df_comp, use_container_width=True)

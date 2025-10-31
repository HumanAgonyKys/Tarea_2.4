import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# -----------------------
# Configuración de la app
# -----------------------
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("🎯 Clustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.title("Leonardo Garcia Bernal 743736")
st.write("""
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
""")

# --- Subir archivo ---
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("⚙️ Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_cols) < 2:
            st.warning("⚠️ Selecciona al menos dos columnas para continuar.")
            st.stop()

        # Parámetros de clustering
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)

        # Visualización PCA
        pca_choice = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # === NUEVOS CONTROLES ===
        st.sidebar.subheader("Parámetros de KMeans")
        init = st.sidebar.selectbox(
            "init",
            options=["k-means++", "random"],
            index=0,
            help="Estrategia de inicialización de centroides."
        )
        max_iter = st.sidebar.number_input(
            "max_iter",
            min_value=1,
            max_value=5000,
            value=300,
            step=50,
            help="Número máximo de iteraciones por ejecución."
        )
        n_init = st.sidebar.number_input(
            "n_init",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Cuántas veces se corre KMeans con diferentes centroides iniciales."
        )
        use_random_state = st.sidebar.checkbox(
            "Fijar random_state",
            value=True,
            help="Actívalo para reproducibilidad."
        )
        if use_random_state:
            random_state = st.sidebar.number_input(
                "random_state",
                min_value=0,
                max_value=1_000_000,
                value=0,
                step=1,
                help="Semilla aleatoria para resultados reproducibles."
            )
        else:
            random_state = None

        # --- Datos y modelo ---
        X = data[selected_cols].copy()

        # Ajuste del número de componentes PCA si el usuario pide 3 pero hay menos features
        n_components_effective = min(pca_choice, len(selected_cols))
        if pca_choice > n_components_effective:
            st.info(f"ℹ️ Cambiando PCA a {n_components_effective} componentes porque seleccionaste {len(selected_cols)} columnas.")

        # Preparar kwargs de KMeans de forma segura
        kmeans_kwargs = dict(
            n_clusters=int(k),
            init=init,
            max_iter=int(max_iter),
            n_init=int(n_init)
        )
        if use_random_state:
            kmeans_kwargs["random_state"] = int(random_state)

        # Entrenar
        kmeans = KMeans(**kmeans_kwargs)
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components_effective)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components_effective)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster'].astype(str)

        # --- Visualización antes del clustering ---
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components_effective == 2:
            # Para mostrar "sin agrupar", pintamos todo igual
            before_df = pca_df.copy()
            before_df["Todos"] = "Datos originales"
            fig_before = px.scatter(
                before_df,
                x='PCA1',
                y='PCA2',
                color="Todos",
                title="Datos originales proyectados con PCA (sin agrupar)",
            )
        else:
            before_df = pca_df.copy()
            before_df["Todos"] = "Datos originales"
            fig_before = px.scatter_3d(
                before_df,
                x='PCA1',
                y='PCA2',
                z='PCA3',
                color="Todos",
                title="Datos originales proyectados con PCA (sin agrupar)",
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- Visualización después del clustering ---
        st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
        if n_components_effective == 2:
            fig_after = px.scatter(
                pca_df,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x='PCA1',
                y='PCA2',
                z='PCA3',
                color='Cluster',
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides ---
        st.subheader("📍 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # --- Método del Codo ---
        st.subheader("📈 Método del Codo (Elbow Method)")
        if st.button("Calcular número óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km_kwargs = dict(
                    n_clusters=i,
                    init=init,
                    max_iter=int(max_iter),
                    n_init=int(n_init)
                )
                if use_random_state:
                    km_kwargs["random_state"] = int(random_state)
                km = KMeans(**km_kwargs)
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(list(K), inertias, 'o-')
            ax2.set_title('Método del Codo')
            ax2.set_xlabel('Número de Clusters (k)')
            ax2.set_ylabel('Inercia (SSE)')
            ax2.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("💾 Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

        # Mostrar parámetros elegidos
        with st.expander("Parámetros usados de KMeans"):
            st.json({**kmeans_kwargs})

else:
    st.info("👉 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |---------------|--------------|------|
    | 45000         | 350          | 28   |
    | 72000         | 680          | 35   |
    | 28000         | 210          | 22   |
    """)

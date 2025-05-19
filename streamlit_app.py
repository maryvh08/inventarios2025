import streamlit as st
import pandas as pd
import numpy as np
import base64

st.title("📊 Sistema de Inventarios con EOQ, ROP e Inventario Actual")

# 📁 Subir archivos CSV
productos_file = st.file_uploader("Sube archivo de Productos (.csv)", type=["csv"])
demandas_file = st.file_uploader("Sube archivo de Demandas mensuales (.csv)", type=["csv"])
inventario_file = st.file_uploader("Sube archivo de Inventario (.csv)", type=["csv"])

if productos_file and demandas_file and inventario_file:
    # 📌 Leer CSVs y limpiar nombres de columnas
    productos_df = pd.read_csv(productos_file)
    productos_df.columns = productos_df.columns.str.strip()

    demandas_df = pd.read_csv(demandas_file)
    demandas_df.columns = demandas_df.columns.str.strip()

    inventario_df = pd.read_csv(inventario_file)
    inventario_df.columns = inventario_df.columns.str.strip()

    # 📌 Verificar columnas necesarias
    columnas_demandas = {'ID_Demanda','ID_Producto','Año','Mes','Cantidad'}
    columnas_productos = {'ID_Producto','Nombre','Unidad_Medida','ID_Proveedor'}
    columnas_inventario = {'ID_Producto','ID_Inventario','Cantidad_Stock','Ubicacion_Almacen','Fecha_Actualizacion'}

    if not columnas_demandas.issubset(demandas_df.columns):
        st.error(f"El archivo de demandas debe tener las columnas: {', '.join(columnas_demandas)}")
    elif not columnas_productos.issubset(productos_df.columns):
        st.error(f"El archivo de productos debe tener las columnas: {', '.join(columnas_productos)}")
    elif not columnas_inventario.issubset(inventario_df.columns):
        st.error(f"El archivo de inventario debe tener las columnas: {', '.join(columnas_inventario)}")
    else:
        # 📌 Calcular demanda promedio y desviación estándar
        resumen = demandas_df.groupby('ID_Producto').agg(
            Demanda_Promedio=('Cantidad', 'mean'),
            Desviacion=('Cantidad', 'std')
        ).reset_index()

        # 📌 Agregar nombre del producto
        resumen = resumen.merge(productos_df, on='ID_Producto', how='left')

        # 📌 Agrupar inventario para obtener el stock total por producto
        inventario_agrupado = inventario_df.groupby('ID_Producto').agg(
            Cantidad_Stock=('Cantidad_Stock', 'sum')
        ).reset_index()

        # 📌 Agregar inventario actual
        resumen = resumen.merge(inventario_agrupado, on='ID_Producto', how='left')

        # 📌 Parámetros fijos
        Z = 1.65          # Nivel de servicio 95%
        Lead_Time = 1     # en meses
        S = 50            # Costo por pedido
        H = 1             # Costo mantenimiento unidad/mes

        # 📌 Cálculos EOQ, SS y ROP
        resumen['SS'] = Z * resumen['Desviacion'] * np.sqrt(Lead_Time)
        resumen['ROP'] = resumen['Demanda_Promedio'] * Lead_Time + resumen['SS']
        resumen['EOQ'] = np.sqrt((2 * resumen['Demanda_Promedio'] * S) / H)

        # 📌 Verificar si hace falta pedido
        resumen['¿Requiere Pedido?'] = np.where(resumen['Cantidad_Stock'] <= resumen['ROP'], '✅ Sí', '❌ No')

        # 📊 Mostrar resultados
        st.subheader("📌 Resultados por Producto:")
        st.dataframe(resumen[['ID_Producto', 'Nombre', 'Demanda_Promedio', 'Desviacion',
                              'SS', 'ROP', 'EOQ', 'Cantidad_Stock', '¿Requiere Pedido?']].round(2))

else:
    st.info("⬆️ Por favor, sube los 3 archivos CSV para continuar.")

# Sidebar con información y opciones
Logo= "Logo.png"
# Leer el archivo como bytes
with open(logo, "rb") as image_file:
    Logo_bytes = image_file.read()

# Codificar a base64 para insertarlo con HTML
logo_base64 = base64.b64encode(Logo_bytes).decode()
with st.sidebar:
    # Logo centrado y redimensionado
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" 
                 alt="Logo" width="300"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    # Acerca de la herramienta
    with st.expander("Acerca de la Herramienta", expanded=False):
        st.markdown("""
        **Inventarya** es una herramienta para realizar calculos de los parámetros de inventario por medio de archivos .csv.
        
        Analiza automáticamente el contenido de los archivos.
        
        La evaluación considera los siguientes aspectos:
        - Demanda Mensual
        - Cantidad en stock
        - Productos
        - Información de adquisición
        
        Versión: 1.0.0
        """)

    st.write("---")


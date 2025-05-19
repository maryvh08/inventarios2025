# 📦 Librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
import math
import tempfile
import os

st.title("📊 Sistema de Inventarios con EOQ y ROP")

# 📁 Cargar archivo Access
archivo = st.file_uploader("Sube el archivo de base de datos Access (.accdb)", type=["accdb"])

if archivo:
    # 🔒 Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".accdb") as tmp:
        tmp.write(archivo.read())
        ruta_temporal = tmp.name

    # 📌 Conexión a la base de datos Access
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        f'DBQ={ruta_temporal};'
    )

    try:
        conexion = pyodbc.connect(conn_str)
        cursor = conexion.cursor()

        # 📌 Consulta de demanda mensual
        query_demanda = """
        SELECT d.ID_Producto, p.Nombre, d.Año, d.Mes, d.Cantidad
        FROM DemandaMensual d
        INNER JOIN Producto p ON d.ID_Producto = p.ID_Producto
        """
        df = pd.read_sql_query(query_demanda, conexion)

        # 📌 Calcular demanda promedio y desviación estándar
        resumen = df.groupby(['ID_Producto', 'Nombre']).agg(
            Demanda_Promedio=('Cantidad', 'mean'),
            Desviacion=('Cantidad', 'std')
        ).reset_index()

        # 📌 Parámetros fijos
        Z = 1.65          # Nivel de servicio del 95%
        Lead_Time = 1     # en meses
        S = 50            # Costo por pedido
        H = 1             # Costo de mantenimiento por unidad por mes

        # 📌 Cálculos EOQ, SS y ROP
        resumen['SS'] = Z * resumen['Desviacion'] * np.sqrt(Lead_Time)
        resumen['ROP'] = resumen['Demanda_Promedio'] * Lead_Time + resumen['SS']
        resumen['EOQ'] = np.sqrt((2 * resumen['Demanda_Promedio'] * S) / H)

        # 📊 Mostrar resultados
        st.subheader("📌 Resultados por Producto:")
        st.dataframe(resumen[['ID_Producto', 'Nombre', 'Demanda_Promedio', 'Desviacion', 'SS', 'ROP', 'EOQ']].round(2))

        # ✅ Cierre de conexión
        conexion.close()
        os.remove(ruta_temporal)

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
else:
    st.info("⬆️ Por favor, sube un archivo .accdb para continuar.")

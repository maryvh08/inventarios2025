import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Sistema de Inventarios con EOQ y ROP")

# 📁 Subir archivo CSV de demanda
archivo = st.file_uploader("Sube archivo CSV con demanda mensual", type=["csv"])

if archivo:
    # 📌 Leer CSV
    df = pd.read_csv(archivo)

    # 📌 Verificar columnas esperadas
    columnas_esperadas = {'ID_Demanda','ID_Producto','Año', 'Mes', 'Cantidad'}
    if not columnas_esperadas.issubset(df.columns):
        st.error(f"El archivo debe tener las columnas: {', '.join(columnas_esperadas)}")
    else:
        # 📌 Calcular demanda promedio y desviación estándar
        resumen = df.groupby(['ID_Producto', 'Nombre']).agg(
            Demanda_Promedio=('Cantidad', 'mean'),
            Desviacion=('Cantidad', 'std')
        ).reset_index()

        # 📌 Parámetros fijos
        Z = 1.65          # Nivel de servicio 95%
        Lead_Time = 1     # en meses
        S = 50            # Costo por pedido
        H = 1             # Costo mantenimiento unidad/mes

        # 📌 Cálculos EOQ, SS y ROP
        resumen['SS'] = Z * resumen['Desviacion'] * np.sqrt(Lead_Time)
        resumen['ROP'] = resumen['Demanda_Promedio'] * Lead_Time + resumen['SS']
        resumen['EOQ'] = np.sqrt((2 * resumen['Demanda_Promedio'] * S) / H)

        # 📊 Mostrar resultados
        st.subheader("📌 Resultados por Producto:")
        st.dataframe(resumen[['ID_Producto', 'Nombre', 'Demanda_Promedio', 'Desviacion', 'SS', 'ROP', 'EOQ']].round(2))

else:
    st.info("⬆️ Por favor, sube un archivo CSV para continuar.")

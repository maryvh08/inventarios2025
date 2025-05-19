# 📦 Librerías necesarias
import streamlit as st
import pyodbc
import pandas as pd
import numpy as np
import math

# 📌 Título de la app
st.title("📦 Control de Inventarios Hospitalarios")

# 📌 Parámetros generales
costo_pedido = 50         # $ por pedido
costo_mantener = 1        # $ por unidad por mes
capacidad_maxima = 10000  # unidades
Z = 1.65                  # Nivel de servicio 95%
lead_time = 1             # meses

# 📌 Ruta base de datos
ruta_bd = st.text_input("📁 Ingresa la ruta de la base de datos Access (.accdb):")

# 📌 Botón para ejecutar
if st.button("📊 Calcular parámetros de inventario"):

    try:
        # 📌 Conexión a la base de datos Access
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            f'DBQ={ruta_bd};'
        )
        conexion = pyodbc.connect(conn_str)

        # 📌 Consulta demanda mensual de enero a abril
        query_demanda_mensual = """
        SELECT d.ID_Producto, p.Nombre, d.Mes, d.Cantidad
        FROM DemandaMensual d
        INNER JOIN Producto p ON d.ID_Producto = p.ID_Producto
        WHERE d.Mes IN ('Enero', 'Febrero', 'Marzo', 'Abril')
        """
        df_demanda_mensual = pd.read_sql_query(query_demanda_mensual, conexion)

        if df_demanda_mensual.empty:
            st.warning("No se encontraron registros de demanda entre enero y abril.")
        else:
            resultados = []

            for producto in df_demanda_mensual['ID_Producto'].unique():
                datos_producto = df_demanda_mensual[df_demanda_mensual['ID_Producto'] == producto]
                nombre = datos_producto['Nombre'].iloc[0]
                demanda_mensual = datos_producto['Cantidad'].values

                D = np.mean(demanda_mensual)  # Demanda promedio mensual
                sigma = np.std(demanda_mensual, ddof=1)  # Desviación estándar
                SS = Z * sigma * math.sqrt(lead_time)  # Stock de seguridad
                ROP = D * lead_time + SS  # Punto de reorden
                EOQ = math.sqrt((2 * D * costo_pedido) / costo_mantener)  # EOQ

                resultados.append({
                    'ID_Producto': producto,
                    'Nombre': nombre,
                    'Demanda_Promedio': round(D, 1),
                    'Desviación_Estandar': round(sigma, 1),
                    'Stock_Seguridad': round(SS),
                    'Punto_Reorden': round(ROP),
                    'EOQ': round(EOQ)
                })

            df_resultados = pd.DataFrame(resultados)

            # 📌 Mostrar tabla de resultados
            st.subheader("📈 Parámetros de inventario calculados")
            st.dataframe(df_resultados)

            # 📌 Validar restricción de capacidad
            espacio_total = df_resultados['EOQ'].sum()

            if espacio_total > capacidad_maxima:
                st.error(f"⚠️ Se requieren {espacio_total:.2f} unidades, pero solo hay {capacidad_maxima} disponibles.")
                factor_ajuste = capacidad_maxima / espacio_total
                df_resultados['EOQ_Ajustado'] = (df_resultados['EOQ'] * factor_ajuste).round()
                st.subheader("📉 EOQ ajustados por restricción de capacidad")
                st.dataframe(df_resultados[['Nombre', 'EOQ', 'EOQ_Ajustado']])
            else:
                st.success(f"✅ Espacio suficiente: {espacio_total:.2f} unidades requeridas de {capacidad_maxima} disponibles.")

        # 📌 Cerrar conexión
        conexion.close()

    except Exception as e:
        st.error(f"❌ Error al conectar con la base de datos o procesar los datos: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tempfile

st.title("📊 Sistema de Inventarios con EOQ, ROP e Inventario Actual desde Base de Datos")

# 📁 Subir archivo de base de datos SQLite
db_file = st.file_uploader("Sube archivo de base de datos SQLite (.db)", type=["db"])

if db_file:
    # 📌 Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(db_file.read())
        db_path = tmp_file.name

    # 📌 Conectar a la base de datos
    conn = sqlite3.connect(db_path)

    # 📌 Verificar que existan las tablas necesarias
    query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
    tables_df = pd.read_sql_query(query_tables, conn)
    tablas = tables_df['name'].tolist()

    tablas_necesarias = ['Productos', 'Demandas', 'Inventario']

    if not set(tablas_necesarias).issubset(set(tablas)):
        st.error(f"La base de datos debe contener las tablas: {', '.join(tablas_necesarias)}")
    else:
        # 📌 Leer datos de las tablas
        productos_df = pd.read_sql_query("SELECT * FROM Productos", conn)
        demandas_df = pd.read_sql_query("SELECT * FROM Demandas", conn)
        inventario_df = pd.read_sql_query("SELECT * FROM Inventario", conn)

        # 📌 Limpiar nombres de columnas
        productos_df.columns = productos_df.columns.str.strip()
        demandas_df.columns = demandas_df.columns.str.strip()
        inventario_df.columns = inventario_df.columns.str.strip()

        # 📌 Verificar columnas necesarias
        columnas_demandas = {'ID_Demanda','ID_Producto','Año','Mes','Cantidad'}
        columnas_productos = {'ID_Producto','Nombre','Unidad_Medida','ID_Proveedor'}
        columnas_inventario = {'ID_Producto','ID_Inventario','Cantidad_Stock','Ubicacion_Almacen','Fecha_Actualizacion'}

        if not columnas_demandas.issubset(demandas_df.columns):
            st.error(f"La tabla Demandas debe tener las columnas: {', '.join(columnas_demandas)}")
        elif not columnas_productos.issubset(productos_df.columns):
            st.error(f"La tabla Productos debe tener las columnas: {', '.join(columnas_productos)}")
        elif not columnas_inventario.issubset(inventario_df.columns):
            st.error(f"La tabla Inventario debe tener las columnas: {', '.join(columnas_inventario)}")
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

    conn.close()

else:
    st.info("⬆️ Por favor, sube el archivo de base de datos SQLite (.db) para continuar.")

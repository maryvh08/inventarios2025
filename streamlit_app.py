import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tempfile

st.title("📊 Sistema de Inventarios con EOQ, ROP e Inventario Actual desde Archivo .SQL")

sql_file = st.file_uploader("Sube archivo SQL (.sql)", type=["sql"])

if sql_file:
    sql_script = sql_file.read().decode('utf-8')

    # Crear archivo temporal para base de datos SQLite
    tmp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp_db_file.name
    tmp_db_file.close()  # Muy importante cerrar aquí para evitar bloqueo

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.executescript(sql_script)
        conn.commit()

        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(tables_query, conn)['name'].tolist()

        tablas_necesarias = ['Productos', 'Demandas', 'Inventario']

        if not set(tablas_necesarias).issubset(set(tables)):
            st.error(f"La base de datos debe contener las tablas: {', '.join(tablas_necesarias)}")
        else:
            productos_df = pd.read_sql_query("SELECT * FROM Productos", conn)
            demandas_df = pd.read_sql_query("SELECT * FROM Demandas", conn)
            inventario_df = pd.read_sql_query("SELECT * FROM Inventario", conn)

            productos_df.columns = productos_df.columns.str.strip()
            demandas_df.columns = demandas_df.columns.str.strip()
            inventario_df.columns = inventario_df.columns.str.strip()

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
                resumen = demandas_df.groupby('ID_Producto').agg(
                    Demanda_Promedio=('Cantidad', 'mean'),
                    Desviacion=('Cantidad', 'std')
                ).reset_index()

                resumen = resumen.merge(productos_df, on='ID_Producto', how='left')

                inventario_agrupado = inventario_df.groupby('ID_Producto').agg(
                    Cantidad_Stock=('Cantidad_Stock', 'sum')
                ).reset_index()

                resumen = resumen.merge(inventario_agrupado, on='ID_Producto', how='left')

                Z = 1.65
                Lead_Time = 1
                S = 50
                H = 1

                resumen['SS'] = Z * resumen['Desviacion'] * np.sqrt(Lead_Time)
                resumen['ROP'] = resumen['Demanda_Promedio'] * Lead_Time + resumen['SS']
                resumen['EOQ'] = np.sqrt((2 * resumen['Demanda_Promedio'] * S) / H)

                resumen['¿Requiere Pedido?'] = np.where(resumen['Cantidad_Stock'] <= resumen['ROP'], '✅ Sí', '❌ No')

                st.subheader("📌 Resultados por Producto:")
                st.dataframe(resumen[['ID_Producto', 'Nombre', 'Demanda_Promedio', 'Desviacion',
                                    'SS', 'ROP', 'EOQ', 'Cantidad_Stock', '¿Requiere Pedido?']].round(2))

    except sqlite3.Error as e:
        st.error(f"Error al ejecutar el script SQL: {e}")
    finally:
        conn.close()

else:
    st.info("⬆️ Por favor, sube el archivo SQL (.sql) para continuar.")

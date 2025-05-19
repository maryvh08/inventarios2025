import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tempfile

st.title("📊 Sistema de Inventarios con EOQ, ROP e Inventario Actual desde Archivo .SQL")

sql_file = st.file_uploader("Sube archivo SQL (.sql)", type=["sql"])

if sql_file:
    # Leer contenido del archivo SQL
    sql_script = sql_file.read().decode('utf-8')

    # Crear base de datos SQLite en archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db_file:
        db_path = tmp_db_file.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Ejecutar el script SQL para crear tablas e insertar datos
        cursor.executescript(sql_script)
        conn.commit()

        # Verificar tablas
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(tables_query, conn)['name'].tolist()

        tablas_necesarias = ['Productos', 'Demandas', 'Inventario']

        if not set(tablas_necesarias).issubset(set(tables)):
            st.error(f"La base de datos debe contener las tablas: {', '.join(tablas_necesarias)}")
        else:
            # Leer tablas
            productos_df = pd.read_sql_query("SELECT * FROM Productos", conn)
            demandas_df = pd.read_sql_query("SELECT * FROM Demandas", conn)
            inventario_df = pd.read_sql_query("SELECT * FROM Inventario", conn)

            # Limpiar columnas
            productos_df.columns = productos_df.columns.str.strip()
            demandas_df.columns = demandas_df.columns.str.strip()
            inventario_df.columns = inventario_df.columns.str.strip()

            # Validar columnas requeridas
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
                # Calcular demanda promedio y desviación estándar
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

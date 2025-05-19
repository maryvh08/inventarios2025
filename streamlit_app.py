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
        columnas_inventario = {'ID_Producto

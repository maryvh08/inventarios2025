import streamlit as st
import sqlite3
import pandas as pd

Link de la herramienta: https://inventarios2025-dnkclvqmpknfzuumjzwnne.streamlit.app/

# Función para conectar a la base de datos
def get_db_connection():
    conn = sqlite3.connect("inventario.db")
    return conn

# Crear tabla si no existe
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventario (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            producto TEXT NOT NULL,
            demanda_estimada INTEGER NOT NULL,
            stock_actual INTEGER NOT NULL,
            inventario_necesario INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Insertar datos en la base de datos
def insertar_datos(producto, demanda_estimada, stock_actual, inventario_necesario):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inventario (producto, demanda_estimada, stock_actual, inventario_necesario)
        VALUES (?, ?, ?, ?)
    ''', (producto, demanda_estimada, stock_actual, inventario_necesario))
    conn.commit()
    conn.close()

# Obtener datos de la base de datos
def obtener_datos():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM inventario", conn)
    conn.close()
    return df

# Interfaz de Streamlit
st.title("Cálculo de Inventario")

# Crear tabla si no existe
create_table()

# Entrada de datos
producto = st.text_input("Nombre del Producto")
demanda_estimada = st.number_input("Demanda Estimada", min_value=0, step=1)
stock_actual = st.number_input("Stock Actual", min_value=0, step=1)

if st.button("Calcular Inventario Necesario"):
    inventario_necesario = max(0, demanda_estimada - stock_actual)
    insertar_datos(producto, demanda_estimada, stock_actual, inventario_necesario)
    st.success(f"Inventario necesario calculado: {inventario_necesario}")

# Mostrar datos almacenados
st.subheader("Registros de Inventario")
df = obtener_datos()
st.dataframe(df)

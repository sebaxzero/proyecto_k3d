# imports

import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from pykrige.ok3d import OrdinaryKriging3D
from pykrige.ok import OrdinaryKriging

# streamlit configuration
st.set_page_config(layout="wide")
st.title("Kriging 3D")
st.sidebar.title("Parametros")
st.sidebar.markdown("Seleccione los parametros para el kriging 3D")
st.sidebar.markdown("")

# lock widget
def callback():
    st.session_state.lock_widget = True

# session state variables
if 'lock_widget' not in st.session_state:
    st.session_state.lock_widget = False

if 'k3d' not in st.session_state:
    st.session_state.k3d = None

if 'k2d' not in st.session_state:
    st.session_state.k2d = None

# plot data function
def plot_data(grid_x, grid_y, grid_z, values):
    fig = px.scatter_3d(x=grid_x.ravel(), y=grid_y.ravel(), z=grid_z.ravel(), 
                        color=values.ravel(), opacity=0.7, title='Kriging 3D Interpolation',
                        color_continuous_scale='Jet', color_continuous_midpoint=np.mean(values))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
                                bgcolor='rgba(0, 0, 0, 0)'))  # Establecer el fondo como transparente
    st.plotly_chart(fig)

def scatter(data1, data2, value, name):
    fig, ax = plt.subplots()
    ax.scatter(data1[x], data1[y], c=data1[value], label=data1[name], cmap='jet')
    ax.scatter(data2[x], data2[y], c=data2[value], label=data2[name], cmap='jet')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)
    
def scatter_krig(gridx, gridy, krig):
    fig, ax = plt.subplots()
    ax.scatter(gridx, gridy, c=krig, cmap='jet')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

# data upload
uploaded_file = st.sidebar.file_uploader("Subir archivo csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.markdown("selección de datos")
    x = st.sidebar.selectbox("Seleccione X", data.columns, index=0, disabled=st.session_state.lock_widget)
    y = st.sidebar.selectbox("Seleccione Y", data.columns, index=1, disabled=st.session_state.lock_widget)
    z = st.sidebar.selectbox("Seleccione Z", data.columns, index=2, disabled=st.session_state.lock_widget)
    v = st.sidebar.selectbox("Seleccione Valor", data.columns, index=3, disabled=st.session_state.lock_widget)
    st.sidebar.markdown("parametros de grilla")
    separationx = st.sidebar.slider(min_value=0.1, max_value=1.0, value=0.1, step=0.1, label="Separacion en X", key='separationx', disabled=st.session_state.lock_widget)
    separationy = st.sidebar.slider(min_value=0.1, max_value=1.0, value=0.1, step=0.1, label="Separacion en Y", key='separationy', disabled=st.session_state.lock_widget)
    separationz = st.sidebar.slider(min_value=0.1, max_value=1.0, value=0.1, step=0.1, label="Separacion en Z", key='separationz', disabled=st.session_state.lock_widget)
    st.sidebar.markdown("")
# main 
def main():
    tab1, tab2, tab3, tab4 = st.tabs(['Datos', 'Modelo 3D', 'Modelo 2D', 'Variograma']) 
    with tab1:
        if uploaded_file:
            plot_data(data[x], data[y], data[z], data[v])
    with tab2:
        if uploaded_file is None:
            st.button(label='Calcular Kriging', key='disabled', disabled=True)
        else:
            datax, datay, dataz, values = data[x], data[y], data[z], data[v]
            rangex, rangey, rangez = [min(datax), max(datax)], [min(datay), max(datay)], [min(dataz), max(dataz)]
            gridx, gridy, gridz = np.arange(rangex[0], rangex[1] + separationx, separationx), np.arange(rangey[0], rangey[1] + separationy, separationy), np.arange(rangez[0], rangez[1] + separationz, separationz)
            if st.button(label='Calcular Kriging', key='enabled', disabled=st.session_state.lock_widget, on_click=callback):
                with st.spinner('Calculando...'):
                    ok3d = OrdinaryKriging3D(x=datax, y=datay, z=dataz, val=values, variogram_model="gaussian")
                    k3d, ss3d = ok3d.execute("grid", gridx, gridy, gridz)
                    st.success('Kriging calculado')
                    time.sleep(1)
                    st.session_state.k3d = k3d
                    st.session_state.lock_widget = False
            
            if st.session_state.k3d is not None:
                with st.spinner('Ploteando...'):
                    st.markdown("## Resultado Kriging 3D")
                    krige = st.session_state.k3d
                    grid_x, grid_y, grid_z = np.meshgrid(gridx, gridy, gridz, indexing='ij')
                    plot_data(grid_x, grid_y, grid_z, krige)
    with tab3:
        if uploaded_file is None:
            st.button(label='Calcular Kriging', key='disabled2d', disabled=True) 
        else:
            hid = st.sidebar.selectbox("seleccione HID", data.columns, index=4, disabled=st.session_state.lock_widget)
            with st.expander("Parametros", expanded=True):
                hid1 = st.selectbox("Seleccione HID", data[hid].unique(), index=0, key='hid1', disabled=st.session_state.lock_widget)
                hid2 = st.selectbox("Seleccione HID", data[hid].unique(), index=1, key='hid2', disabled=st.session_state.lock_widget)
                hid1_data, hid2_data = data[data[hid] == hid1], data[data[hid] == hid2]
                rangex2d, rangey2d = [min(min(hid1_data[x]), min(hid2_data[x])), max(max(hid1_data[x]), max(hid2_data[x]))], [min(min(hid1_data[y]), min(hid2_data[y])), max(max(hid1_data[y]), max(hid2_data[y]))]
                gridx_2d, gridy_2d = np.arange(rangex2d[0], rangex2d[1] + separationx, separationx), np.arange(rangey2d[0], rangey2d[1] + separationy, separationy)
                scatter(hid1_data, hid2_data, v, hid)
            if st.button(label='Calcular Kriging', key='enabled2d', disabled=st.session_state.lock_widget, on_click=callback):
                with st.spinner('Calculando...'):
                    x2d = np.concatenate((hid1_data[x], hid2_data[x]))
                    y2d = np.concatenate((hid1_data[y], hid2_data[y]))
                    val2d = np.concatenate((hid1_data[v], hid2_data[v]))
                    ok2d = OrdinaryKriging(x=x2d, y=y2d, z=val2d, variogram_model="gaussian")
                    k2d, ss2d = ok2d.execute("grid", gridx_2d, gridy_2d)
                    st.success('Kriging calculado')
                    time.sleep(1)
                    st.session_state.k2d = k2d
                    st.session_state.lock_widget = False
                    
            if st.session_state.k2d is not None:
                with st.spinner('Ploteando...'):
                    st.markdown("## Resultado Kriging 2D")
                    krige_2d = st.session_state.k2d
                    grid_x_2d, grid_y_2d = np.meshgrid(gridx_2d, gridy_2d, indexing='ij')
                    scatter_krig(grid_x_2d, grid_y_2d, krige_2d)
    with tab4:
        if st.session_state.k3d is not None:
            st.write(k3d)
            st.write(ss3d)    
                          
if __name__ == "__main__":
    main()
        
#todo
# - agregar variograma
# - agregar exportar a csv
# - agregar guardar imagen

#corregir  
# - kriging 2d no funciona correctamente (aparentemente, hay que comprobar)
##################################

# IMPORTANCION DE LIBRERÍAS

##################################

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import warnings
warnings.simplefilter("ignore")

##################################

# PRE-PROCESAMIENTO DE LOS DATOS

##################################

# Leer dataset
df = pd.read_csv("C:/workspace/Reto/Datos/iar_Reservaciones.csv")

# Eliminar columnas con valores confidenciales
df.drop(['h_correo_e', 'h_nom'], axis = 'columns', inplace = True)

# Sustituir valores 'vacios' (espacios) con nulos
df.replace(r'^\s*$', np.nan, regex=True, inplace = True)

# Formatear correctamente las fechas que se usaran
columnas_fechas = ['h_res_fec', 'h_fec_lld']
for fecha in columnas_fechas:
    df[fecha] = pd.to_datetime(df[fecha], format='%Y%m%d', errors='coerce')

# Eliminar columnas por tener muchos valores nulos
df.drop(columns=['h_cod_reserva', 'h_codigop'], axis='columns', inplace = True)

# Eliminar registros que tienen algun campo con valor nulo
df.dropna(inplace = True)

# Quedarnos solo con las columnas que utilizaremos
cols_relevantes = ['ID_Reserva', 'ID_estatus_reservaciones', 'ID_Agencia', 'ID_Paquete', 'ID_Tipo_Habitacion', 
                   'h_tot_hab', 'h_num_men', 'h_num_adu', 'h_num_per', 
                   'h_res_fec', 'h_fec_lld', 'h_num_noc',
                   'h_tfa_total']
df = df[cols_relevantes]

# Extraer el mes de las fechas que se usaran
for fecha in columnas_fechas:
    df[f'{fecha}_mes'] = df[fecha].dt.strftime("%B").astype("category")

# Establecer indica
df.set_index(df['ID_Reserva'], inplace = True)
df.drop(['ID_Reserva'], axis = 'columns', inplace = True)

# Sustituir todos los valores de no. de personas por la suma de no. de menores y no. de adultos
df.loc[df['h_num_per'], 'h_num_per'] = df['h_num_men'] + df['h_num_adu']

# Crear nuevas columnas indicando si es fin de semana o entre semana
fechas = ['reservacion', 'estadia']

for i in range(len(fechas)):
    df[f'entre_fin_{fechas[i]}'] = df[columnas_fechas[i]].dt.weekday.apply(lambda x: 'Fin de semana' if x >= 5 else 'Entre semana')


# Para reducir la cardinalidad de las columnas, quedándonos con el porcentaje del umbral 
# y agrupando lo restante en una categoría nueva llamada "Otro"
umbral = 0.90
cols_reduccion_cardinalidad = ['ID_Agencia', 'ID_Tipo_Habitacion']

for col in cols_reduccion_cardinalidad:
    freqs = df[col].value_counts(normalize=True)
    top_cats = freqs.cumsum()[freqs.cumsum() <= umbral].index
    df[col + '_reducida'] = df[col].where(df[col].isin(top_cats), other='Otro')

df.drop(columns = cols_reduccion_cardinalidad, inplace = True)
df.columns = [col.replace('_reducida', '') for col in df.columns]

# Conservar solamente los registros en los que la tarifa total y el no. de noches es diferente de 0
df = df[(df['h_tfa_total'] != 0) & (df['h_num_noc'] != 0)]

# Generar una columna nueva para tener la tarifa por noche
df['tarifa_x_noche'] = df['h_tfa_total'] / df['h_num_noc']

##################################

# GENERAR EL DATASET PARA K-MODES

##################################

# Esta primera columna se usa para crear el nuevo dataframe de valores categoricos
df_categ = pd.DataFrame(df['ID_estatus_reservaciones'].astype(str))

# Conservamos el indice del dataframe original para el analisis de clústers
df_categ = df_categ.set_index(df.index)  

# Columnas que se agregarán al nuevo dataframe de valores categóricos y no requieren ningún tipo de transformación
Columns_to_categ = ['h_tot_hab', 'ID_Paquete', 'h_num_men', 'h_num_adu', 'h_num_per',
                    'entre_fin_reservacion', 'ID_Agencia', 'ID_Tipo_Habitacion', 'h_res_fec_mes', 'h_fec_lld_mes']

for column in Columns_to_categ:
    df_categ[column] = df[column].astype(str)

# Dividir los registros categoricamente por cuartiles (cantidades iguales de valores por rango),
# basados en la tarifa por noche
q1 = df['tarifa_x_noche'].quantile(0.25)
q2 = df['tarifa_x_noche'].quantile(0.50)
q3 = df['tarifa_x_noche'].quantile(0.75)

tarifa_categ = []
for tarifa in df['tarifa_x_noche']:
    if tarifa <= q1:
        tarifa_categ.append('Bajo')
    elif tarifa > q1 and tarifa <= q2:
        tarifa_categ.append('Medio-Bajo')
    elif tarifa > q2 and tarifa <= q3:
        tarifa_categ.append('Medio-Alto')
    else:
        tarifa_categ.append('Alto')
    
df_categ['tarifa_x_noche'] = tarifa_categ
df_categ['tarifa_x_noche'] = df_categ['tarifa_x_noche'].astype(str)

##################################

# APLICAR ANALISIS K-MODES

##################################

# Aplicar K-Modes con K=4
km = KModes(n_clusters=4, init="Huang", n_init=5, verbose=0, random_state=42)
clusters = km.fit_predict(df_categ)

# Agregar la columna de clusters al DataFrame
df_categ['cluster'] = clusters

# añadimos los resultados al dataframe completo
df = df.join(df_categ[['cluster']], how='inner')

##################################

# EXTRACCION DE INDICADORES PARA
# EL DASHBOARD Y LAS 
# RECOMENDACIONES PERSONALIZADAS

##################################

# Columnas a las que se le aplicara el analisis numerico
numericas = ['h_tot_hab', 'tarifa_x_noche', 'h_num_per', 'h_num_adu', 'h_num_men', 'h_num_noc']

# Diccionario en el que guardarán los resultados por columna por cluster 
# (el arreglo 0 será media y 1 será desv. est.)
resultados = {}
for valor in numericas:
    medias = df.groupby('cluster')[valor].agg(['mean']).to_numpy().flatten()
    medias = np.round(medias, 2)
    desvests = df.groupby('cluster')[valor].agg(['std']).to_numpy().flatten()
    desvests = np.round(desvests, 2)
    resultados[valor] = []
    resultados[valor].append(medias)
    resultados[valor].append(desvests)

# Funcion para extraer el valor valor maximo, cluster al que corresponde, 
# y diferencia porcentual con el segundo maximo valor (aplica a las medias)
def resumen_maximo(arr):
    max_val = np.max(arr)
    max_idx = np.argmax(arr)
    arr_sin_max = np.delete(arr, max_idx)
    segundo_max = np.max(arr_sin_max)
    diferencia_pct = 100 * (max_val - segundo_max) / max_val
    return max_val, max_idx, diferencia_pct

# Lo mismo pero con el minimo (aplica a las desvests)
def resumen_minimo(arr):
    min_val = np.min(arr)
    min_idx = np.argmin(arr)
    arr_sin_min = np.delete(arr, min_idx)
    segundo_min = np.max(arr_sin_min)
    diferencia_pct = 100 * (min_val - segundo_min) / min_val
    return min_val, min_idx, diferencia_pct

# Funcion que regresa un dataframe con el porcentaje de presencia de cada categoria agrupando 
# por otra (para aplicarse a mes de reservacion y de estadia agrupando por cluster)
def porcentaje_por_categoria(df, columna_grupo, columna_valor):
    conteos = df.groupby(columna_grupo)[columna_valor].value_counts(normalize=True)
    porcentaje_df = conteos.rename("porcentaje").reset_index()
    porcentaje_df["porcentaje"] *= 100
    porcentaje_df["porcentaje"] = np.round(porcentaje_df["porcentaje"], 2)
    return porcentaje_df

##################################

# LOGICA PARA RECOMENDACIONES 
# PERSONALIZADAS

##################################

# Importamos librerías
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Importamos los datos contenidos en Team.csv
data = pd.read_csv('Teams.csv')
print(data)

# **************************************
# *****   ANÁLISIS DE LOS DATOS    *****
# **************************************
# Analizamos los datos
print(data.shape)
print(data.head(15))

# Formato de los datos
print(data.dtypes)
# Descripción de los datos
print(data.describe())
# Buscamos datos nulos o faltantes
print(data.isnull().sum())


# **************************************
# ***** PROCESAMIENTO DE LOS DATOS *****
# **************************************

# Eliminación de columnas innecesarias
borrar_columnas = ['Liga','Franquicia','Clasificacion','Local',
                   'Gan_Division','Gan_WC','Gan_Liga','Gan_WS',
                   'Sacrificados','Nombre_Equipo','Nombre_Estadio',
                   'Asistencia_Estadio','Factor_Bateadores',
                   'Factor_Pitcher','Id_R','Id_BRW','Id_RW']
data = data.drop(borrar_columnas, axis=1)
data = data.drop(['Eliminados','Eliminados_Pitcher','Division'], axis=1)

# Completamos los datos nulos con la media de cada uno
data['Carreras']        = data['Carreras'].fillna(data['Carreras'].median())
data['Strike_Fuera']    = data['Strike_Fuera'].fillna(data['Strike_Fuera'].median())
data['Bases_Robadas']   = data['Bases_Robadas'].fillna(data['Bases_Robadas'].median())

# Analizamos y visualizamos nuestros datos después transformaciones
print(data.shape)
print(data.head(15))
# Formato de los datos
print(data.dtypes)
# Descripción de los datos
print(data.describe())
# Buscamos datos nulos o faltantes
print(data.isnull().sum())

# Veamos correlaciones mediante algunos coeficientes
#   Coeficiente de pearson +- >0.75 descartado
pearson     = data.corr(method="pearson")
print(pearson)
plt.figure(figsize=(14, 8))
pearson.style.background_gradient(cmap='coolwarm')
sns.heatmap(pearson, annot=True)
plt.title("CORRELACIONES MEDIANTE COEFICIENTE DE PEARSON")
plt.tight_layout()
plt.show()

#   Coeficiente de spearman +- >0.75 descartado
spearman    = data.corr(method= "spearman", min_periods = 4)
print(spearman)
plt.figure(figsize=(14, 8))
spearman.style.background_gradient(cmap='coolwarm')
sns.heatmap(spearman, annot=True)
plt.title("CORRELACIONES MEDIANTE COEFICIENTE DE SPEARMAN")
plt.tight_layout()
plt.show()

# Otenemos la asimetría o sesgo
print(data.skew(skipna = True , level = None , numeric_only = None))

# Seleccionamos un equipo para evaluar
equipo = 'NYA'
data_equipo = data.loc[data.Equipo == equipo]

#   Creamos dataframes por característica y visulizamos en grupos de 9(3x3)
p_jugados           = data_equipo["P_Jugados"]
ganados             = data_equipo["Ganados"]
perdidos            = data_equipo["Perdidos"]
carreras_ganadas    = data_equipo["Carreras_Ganadas"]
bateos              = data_equipo["Bateos"]
golpes_bateador     = data_equipo["Golpes_Bateador"]
dobles              = data_equipo["Dobles"]
triples             = data_equipo["Triples"]
home_runs           = data_equipo["Home_Runs"]
#   Visualizamos las distribuciones
fig = plt.figure(figsize=(12,6))
ax_p_jugados        = fig.add_subplot(3,3,1)
ax_ganados          = fig.add_subplot(3,3,2)
ax_perdidos         = fig.add_subplot(3,3,3)
ax_carreras_ganadas = fig.add_subplot(3,3,4)
ax_bateos           = fig.add_subplot(3,3,5)
ax_golpes_bateador  = fig.add_subplot(3,3,6)
ax_dobles           = fig.add_subplot(3,3,7)
ax_triples          = fig.add_subplot(3,3,8)
ax_home_runs        = fig.add_subplot(3,3,9)

ax_p_jugados.hist(p_jugados,        bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_p_jugados.set_xlabel("PARTIDOS JUGADOS {}".format(equipo))
ax_p_jugados.set_title("HISTOGRAMA PARTIDOS JUGADOS {}".format(equipo))

ax_ganados.hist(ganados,            bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_ganados.set_xlabel("PARTIDOS GANADOS {}".format(equipo))
ax_ganados.set_title("HISTOGRAMA PARTIDOS GANADOS {}".format(equipo))

ax_perdidos.hist(perdidos,          bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_perdidos.set_xlabel("PARTIDOS PERDIDOS {}".format(equipo))
ax_perdidos.set_title("HISTOGRAMA PARTIDOS PERDIDOS {}".format(equipo))

ax_carreras_ganadas.hist(carreras_ganadas,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_carreras_ganadas.set_xlabel("CARRERAS GANADAS {}".format(equipo))
ax_carreras_ganadas.set_title("HISTOGRAMA CARRERAS GANADAS {}".format(equipo))

ax_bateos.hist(bateos,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_bateos.set_xlabel("BATEOS {}".format(equipo))
ax_bateos.set_title("HISTOGRAMA BATEOS REALIZADOS {}".format(equipo))

ax_golpes_bateador.hist(golpes_bateador,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_golpes_bateador.set_xlabel("GOLPES BATEADOR {}".format(equipo))
ax_golpes_bateador.set_title("HISTOGRAMA GOLPES BATEADOR {}".format(equipo))

ax_dobles.hist(dobles,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_dobles.set_xlabel("DOBLES CARRERAS {}".format(equipo))
ax_dobles.set_title("HISTOGRAMA DOBLES CARRERAS {}".format(equipo))

ax_triples.hist(triples,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_triples.set_xlabel("TRIPLES CARRERAS {}".format(equipo))
ax_triples.set_title("HISTOGRAMA TRIPLES CARRERAS {}".format(equipo))

ax_home_runs.hist(home_runs,   bins=10,    color='green',  linewidth=1,    edgecolor="black")
ax_home_runs.set_xlabel("HOME RUNS {}".format(equipo))
ax_home_runs.set_title("HISTOGRAMA HOME RUNS {}".format(equipo))

plt.tight_layout()
plt.show()


# Separamos por eras o etapas
i =0
for year in data['Agno']:
    if year < 1920:
        data.loc[i,"era"]=1
    elif year >= 1920 and year <= 1941:
        data.loc[i, "era"] = 2
    elif year >= 1942 and year <= 1945:
        data.loc[i, "era"] = 3
    elif year >= 1946 and year <= 1962:
        data.loc[i, "era"] = 4
    elif year >= 1963 and year <= 1976:
        data.loc[i, "era"] = 5
    elif year >= 1977 and year <= 1992:
        data.loc[i, "era"] = 6
    elif year >= 1993 and year <= 2009:
        data.loc[i, "era"] = 7
    elif year >= 2010:
        data.loc[i, "era"] = 8
    i += 1

# Separamos por décadas
j =0
for year in data['Agno']:
    if year     <   1920:
        data.loc[j,"decada"]    =   1910

    elif year   >=  1920 and year   <=  1929:
        data.loc[j, "decada"]   =   1920

    elif year   >=  1930 and year   <=  1939:
        data.loc[j, "decada"]   =   1930

    elif year   >=  1940 and year   <=  1949:
        data.loc[j, "decada"]   =   1940

    elif year   >=  1950 and year   <=  1959:
        data.loc[j, "decada"]   =   1950

    elif year   >=  1960 and year   <=  1969:
        data.loc[j, "decada"]   =   1960

    elif year   >=  1970 and year   <=  1979:
        data.loc[j, "decada"]   =   1970

    elif year   >=  1980 and year   <=  1989:
        data.loc[j, "decada"]   =   1980

    elif year   >=  1990 and year   <=  1999:
        data.loc[j, "decada"]   =   1990

    elif year   >=  2000 and year   <=  2009:
        data.loc[j, "decada"]   =   2000

    elif year   >=  2010:
        data.loc[j, "decada"]   =   2010

    j += 1
print(data.Equipo)

# Seleccionamos un equipo para evaluar
equipo = 'NYA'
data_equipo = data.loc[data.Equipo == equipo]
equipo1 = 'TEX'
data_equipo1 = data.loc[data.Equipo == equipo1]

# **************************************
# *****   VISUALIZACIÓN GRÁFICA    *****
# **************************************
# Graficamos datos
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.hist(data_equipo['Ganados'],bins=10, color='green',linewidth=1, edgecolor="black")
ax1.set_xlabel("JUEGOS GANADOS EQUIPO {}".format(equipo))
ax1.set_title("HISTOGRAMA JUEGOS GANADOS EQUIPO {}".format(equipo))

ax2.hist(data_equipo['Perdidos'],bins=10, color='red', linewidth=1, edgecolor="black")
ax2.set_xlabel("JUEGOS PERDIDOS EQUIPO {}".format(equipo))
ax2.set_title("HISTOGRAMA JUEGOS PERDIDOS EQUIPO {}".format(equipo))

ax3.hist(data_equipo1['Ganados'],bins=10, color='green', linewidth=1, edgecolor="black")
ax3.set_xlabel("JUEGOS GANADOS EQUIPO {}".format(equipo1))
ax3.set_title("HISTOGRAMA JUEGOS GANADOS EQUIPO {}".format(equipo1))

ax4.hist(data_equipo1['Perdidos'],bins=10, color='red',linewidth=1, edgecolor="black")
ax4.set_xlabel("JUEGOS PERDIDOS EQUIPO {}".format(equipo1))
ax4.set_title("HISTOGRAMA JUEGOS PERDIDOS EQUIPO {}".format(equipo1))
plt.tight_layout()
plt.show()

# Graficamos las carreras realizadas
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(data_equipo['Agno'],data_equipo['Carreras'],linewidth=2.0,color="green")
ax1.set_title("CARRERAS REALIZADAS / AÑO - EQUIPO: {}".format(equipo))
ax1.set_ylabel("CARRERAS REALIZADAS")
ax1.set_xlabel("AÑO")
ax2.plot(data_equipo1['Agno'],data_equipo1['Carreras'],linewidth=2.0,color="green")
ax2.set_title("CARRERAS REALIZADAS / AÑO - EQUIPO: {}".format(equipo1))
ax2.set_xlabel("AÑO")
ax2.set_ylabel("CARRERAS REALIZADAS")
plt.tight_layout()
plt.show()

# Gráficas adicionales
CarrerasG_x_Partido  = data_equipo['Carreras_Ganadas'] / data_equipo['P_Jugados']
CarrerasG_x_Partido1 = data_equipo1['Carreras_Ganadas'] / data_equipo1['P_Jugados']
CarrerasP_x_Partido = data_equipo['Carreras_Oponentes'] / data_equipo['P_Jugados']
CarrerasP_x_Partido1 = data_equipo1['Carreras_Oponentes'] / data_equipo1['P_Jugados']

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.scatter(CarrerasG_x_Partido,data_equipo['Ganados'], c="green")
ax1.set_title("Carreras por Juegos vs. Juegos Ganados")
ax1.set_ylabel("Juegos Ganados")
ax1.set_xlabel("Carreras por Juego - Equipo: {}".format(equipo))
ax2.scatter(CarrerasP_x_Partido,data_equipo['Ganados'], c="red")
ax2.set_title("Carreras Permitidas por Juegos vs. Juegos Ganados")
ax2.set_ylabel("Juegos Ganados")
ax2.set_xlabel("Carreras Permitidas por Juego - Equipo: {}".format(equipo))
ax3.scatter(CarrerasG_x_Partido1,data_equipo1['Ganados'], c="green")
ax3.set_title("Carreras por Juegos vs. Juegos Ganados")
ax3.set_ylabel("Juegos Ganados")
ax3.set_xlabel("Carreras por Juego - Equipo: {}".format(equipo1))
ax4.scatter(CarrerasP_x_Partido1,data_equipo1['Ganados'], c="red")
ax4.set_title("Carreras Permitidas por Juegos vs. Juegos Ganados")
ax4.set_ylabel("Juegos Ganados")
ax4.set_xlabel("Carreras Permitidas por Juego - Equipo: {}".format(equipo1))
plt.tight_layout()
plt.show()

# Eliminamos últimas columnas que no son necesarias
data_equipo = data_equipo.drop(['Agno','Equipo'], axis = 1)
data_equipo1 = data_equipo1.drop(['Agno','Equipo'], axis = 1)

# Importamos nuevas librerías para entrenar nuestro modelo
from sklearn.model_selection    import train_test_split
from sklearn.linear_model       import LinearRegression
from sklearn.metrics            import r2_score
from sklearn.metrics            import mean_squared_error

# Definimos las variables independientes y la dependiente
X = data_equipo.drop("Ganados",axis = 1)
y = data_equipo["Ganados"]
X1 = data_equipo1.drop("Ganados",axis = 1)
y1 = data_equipo1["Ganados"]

# Separamos entre entrenamiento y prueba
X_train, X_test, y_train, y_test        = train_test_split(X, y, test_size=0.20, random_state=1)
X_train1, X_test1, y_train1, y_test1    = train_test_split(X1, y1, test_size=0.20, random_state=1)

# Elegimos el modelo "Regresión Lineal"
algoritmo = LinearRegression()
algoritmo1 = LinearRegression()

# Entrenamos el algoritmo
algoritmo.fit(X_train, y_train)
algoritmo1.fit(X_train1, y_train1)

# Realizamos una predicción
y_test_pred = algoritmo.predict(X_test)
y_test_pred1 = algoritmo1.predict(X_test1)

# Calculamos la precisión del modelo
# Error promedio al cuadrado
# Calculo de R2
mse =mean_squared_error(y_test,y_test_pred)
rmse_rf =(mean_squared_error(y_test,y_test_pred))**(1/2)
r2 = r2_score(y_test,y_test_pred)
mse1 =mean_squared_error(y_test1,y_test_pred1)
rmse_rf1 =(mean_squared_error(y_test1,y_test_pred1))**(1/2)
r21 = r2_score(y_test1,y_test_pred1)
print("****************************************************************")
print("**********       MSE {}:                    {}".format(equipo, mse))
print("**********       ERROR CUADRÁTICO MEDIO {}: {}".format(equipo, rmse_rf))
print("**********       R2 {}:                     {}".format(equipo, r2))
print("**********       MSE {}:                    {}".format(equipo1, mse1))
print("**********       ERROR CUADRÁTICO MEDIO {}: {}".format(equipo1, rmse_rf1))
print("**********       R2 {}:                     {}".format(equipo1, r21))
print("****************************************************************")

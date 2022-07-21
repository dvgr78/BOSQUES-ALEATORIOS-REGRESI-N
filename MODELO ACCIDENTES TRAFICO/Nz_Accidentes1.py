# Importaremos las librerías necesarias para el proyecto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importaremos nuestros datos a un dataframe de pandas
data = pd.read_csv('Crash_Analysis_System_CAS_data.csv')

# **************************************
# ********* ANALIZAR LOS DATOS *********
# **************************************

# Primera vista general de los datos importados
print(data)
# para conocer la estructura de los datos
print(data.shape)
# para conocer el formato de los datos
print(data.dtypes)
# Para conocer los datos nulos
print(data.isnull().sum())

# **************************************
# ***** PROCESAMIENTO DE LOS DATOS *****
# **************************************

# Eliminación de columnas innecesarias
data = data.drop(['X','Y','OBJECTID_1','OBJECTID'], axis = 1)
# Eliminación de columnas a las que le faltan datos
data = data.drop(['crashDirec','crashRPDir','crashRPSH','crashRPNew','trafficCon','roadLane','cornerRoad'], axis = 1)
# Reducimos el conjunto de datos y trabajaremos con los informados a partir del año 2010
data = data.loc[(data['crashYear']>2010), :]

print(data)
print(data.shape)
print(data.dtypes)
print(data.isnull().sum())

# **************************************
# ***** VISUALIZACIÓN DE LOS DATOS *****
# **************************************

# Obtenemos el valor del año más bajo
inicio  = data.crashYear.min()
# Obtenemos el valor del año más alto
final   = data.crashYear.max()
# Sumamos la cantidad de "muertes"
muertes = data['fatalCount'].sum()
# Sumamos la cantidad de "heridos graves"
graves  = data['seriousInj'].sum()
# Sumamos la cantidad de "heridos leves"
leves   = data['minorInjur'].sum()

print("***********************************************************************")
print("***************             DATOS RELEVANTES            ***************")
print("***********************************************************************")
print("**** Resultados desde el año:        {}".format(inicio)," hasta el año:  {}".format(final))
print("**** Muertes hasta el año {}:      {}".format(final,muertes))
print("**** Heridos graves:                {}".format(graves))
print("**** Herdidos leves:                {}".format(leves))
print("***********************************************************************")

# **************************************
# *****   VISUALIZACIÓN  GRÁFICA   *****
# **************************************

# Graficamos el número de muertes por año
g_muertes = sns.barplot(x= "crashYear", y ="fatalCount",data=data)
g_muertes.set(xlabel="AÑO", ylabel = "MUERTES EN MILES")
g_muertes.set_title('Muertes / Año')
plt.show()
# Graficamos el número de heridos graves por año
g_graves = sns.barplot(x= "crashYear", y ="seriousInj", data=data)
g_graves.set(xlabel="AÑO", ylabel = "HERIDOS GRAVES EN MILES")
g_graves.set_title('Heridos Graves / Año')
plt.show()
# Graficamos el número de heridos leves por año
g_leves = sns.barplot(x= "crashYear", y ="minorInjur", data=data)
g_leves.set(xlabel="AÑO", ylabel = "HERIDOS LEVES EN MILES")
g_leves.set_title('Heridos Leves / Año')
plt.show()
# Graficamos cantidad según gravedad de heridas de forma horizontal
# N: sin daños
# M: leves
# S: graves
# F: muertes
g_gravedad = sns.countplot(y ="crashSever", data=data, order="NMSF")
g_gravedad.set(xlabel="CANTIDAD", ylabel = "GRAVEDAD LESIONES")
g_gravedad.set_title('CANTIDAD / GRAVEDAD LESIONES')
plt.show()

# PARA PSAR LOS OBJETOS A DATOS
# importamos nueva librería
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()                        # convierte datos repetitivos en números
# Evaluamos cada una de las columnas para encodar tipos objetos
for i in data:
    if data[i].dtype == 'object':
        encoder.fit(data[i].astype(str))
        data[i]= encoder.transform(data[i])
data = pd.get_dummies(data)

# **************************************
# *****  ENTRENAMIENTO DEL MODELO  *****
# **************************************

# importamos nuevas librerías para entrenar nuestro modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Definimos los datos dependientes e independientes
X = data.drop('fatalCount', axis=1)
y = data['fatalCount']

# Seperamos entre datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)
# Definimos el algoritmo a utilizar "Bosques Aleatorios Regresión"
algoritmo = RandomForestRegressor(n_estimators=60)
# Entrenamos el algortimo
algoritmo.fit(X_train, y_train)
# Realizamos una predicción
y_test_pred = algoritmo.predict(X_test)
# Calculamos la precisión del modelo con R2
print("***************************************************************")
print("**** PRECISIÓN DEL MODELO R2: ",r2_score(y_test,y_test_pred),"         ****")
print("***************************************************************")

















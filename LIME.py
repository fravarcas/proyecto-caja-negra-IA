import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


adults = pd.read_csv('adult.data', header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education',
                              'education-num', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'money'])

attributes = adults.loc[:, 'age':'native-country']
goal = adults['money']

codificator_attributes = sk.preprocessing.OrdinalEncoder()
codificator_attributes.fit(attributes)
codified_attributes = codificator_attributes.transform(attributes)

codificator_goal = sk.preprocessing.LabelEncoder()
codified_goal = codificator_goal.fit_transform(goal)

(training_attributes, test_attributes,
 training_goal, test_goal) = model_selection.train_test_split(
     codified_attributes,codified_goal, random_state=12345, test_size=.33, stratify=codified_goal)

#Entrenamiento de modelo random forest model
randomForestModel = RandomForestClassifier(n_estimator=100, max_depth=10, random_state=42)
randomForestModel.fit(training_attributes, training_goal)

max_attributes = [max(attributes[x][:]) for x in range(len(attributes))]
min_attributes = [min(attributes[x][:]) for x in range(len(attributes))]


def cosine_distance(vector1, vector2):

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_distance = dot_product / (norm1 * norm2)

    return cosine_distance


def LIMEAlgorithm(data, f, N, max_attributes, min_attributes):

    #Inicalizamos las listas donde se guardaran los diferentes conjuntos
    X = []
    R = []
    W = []
    Y = []

    #Realizamos N iteraciones para generar N permutaciones de los atributos de la muestra
    for i in range (1, N):
    
    #Generamos una lista aleatoria que tenga como maximo el tamaño de la muestra y cuyos enteros esten entre 0 y el tamaño de la muestra
        attributes = np.random.choice(np.random.randint(len(data)),np.random.randint(len(data)))
    #Generamos las sublistas que representaran los diferentes valores para 1 permutación de la muestra
        element_R = []
        element_W = []
        element_X = []
    #Recorremos los atributos de la muestra
        for j in range(len(data)):
    #Comprobamos si el indice del atributo está en la lista generada aleatoriamente para decidir si perturbamos el atributo o no
            if j in attributes:

                element_R.append(0)
                modifiedAttribute = np.random.randint(min_attributes[j], max_attributes[j])
                element_X.append(modifiedAttribute)

            else:

                element_R.append(1)
                element_X.append(data[j])
            

            element_W.append(cosine_distance(element_X, data))

            X.append(element_X)
            R.append(element_R)
            W.append(element_W)
    #Una vez realizadas las permutaciones las ponderamos con los pesos generados en W
    R_ponderada = []

    for i in range(len(R)):

        num = W[i]
        sublist = R[i]
        R_muestra_ponderada = []

        for j in range(len(sublist)):

            res = num * sublist[j]
            R_muestra_ponderada.append(res)

        R_ponderada.append(R_muestra_ponderada)

    #Realizamos las predicciones de las permutaciones con el modelo pasado como parametro
    for i in range(len(X)):

        prediction = f.predict(X[i].to_numpy())
        Y.append(prediction)

    #Se entrena el modelo ridge pasandole las R ponderadas como atributos y las predicciones de Y como objetivo
    G = Ridge(alpha=1.0)
    G.fit(R_ponderada, Y)

    return G.get_params()

#Metricas

def identidad(muestra_1, muestra_2):

    cosine_distance(muestra_1, muestra_2)
    if cosine_distance == 0:
        print('estas muestras son identicas')
    
    else:
        print('estas muestras no son identicas')
        
        
def coherencia(p,e):
    diferencia=abs(p-e)
    return diferencia


def completitud(error_explicacion, error_prediccion):
    return error_explicacion / error_prediccion

completitud_resultado = completitud(error_explicacion, error_prediccion)
porcentaje_completitud = completitud_resultado * 100

print(f"La completitud es: {porcentaje_completitud}")


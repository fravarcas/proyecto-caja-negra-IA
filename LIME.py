import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
from sklearn.linear_model import Ridge

adults = pd.read_csv('adult.data', header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education',
                              'education-num', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'money'])

attributes = adults.loc[:, 'age':'native-country']
goal = adults['money']

codificator_attributes = sk.preprocessing.OrdinalEncoder()
codificator_attributes.fit(attributes)
codified_attributes = codificator_attributes.transform(attributes)


def cosine_distance(vector1, vector2):

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_distance = dot_product / (norm1 * norm2)

    return cosine_distance


def LIMEAlgorithm(data, f, N, max_attributes, min_attributes):

    X = []
    R = []
    W = []

    for i in range (1, N):

        attributes = np.random.choice(np.random.randint(1, f.size()),np.random.randint(1, f.size()))
        element_R = []
        element_W = []
        element_X = []

        for j in range(0, data.size() - 1):

            if j in attributes:

                element_R.append(0)
                modifiedAttribute = np.random.randint(min_attributes[j], max_attributes[j])
                element_X.append(modifiedAttribute)

            else:

                element_R.append(1)
                element_X.append(data[j])
            

            element_W = element_W + (cosine_distance(element_X, data))

            X.append(element_X)
            R.append(element_R)
            W.append(element_W)

    R_ponderada = []

    for sublist1, sublist2 in zip(R, W):
        sublist_result = []
        for num1, num2 in zip(sublist1, sublist2):
            sublist_result.append(num1 * num2)
        R_ponderada.append(sublist_result)

    Y = [f(x) for x in X]

    G = Ridge(alpha=1.0)
    G.fit(R_ponderada, Y)

    return 



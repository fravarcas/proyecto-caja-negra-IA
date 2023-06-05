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
        tuple_R = tuple()
        tuple_W = tuple()
        tuple_X = tuple()

        for j in range(0, data.size()):

            if j in attributes:

                tuple_R = tuple_R + (0)
                modifiedAttribute = np.random.randint(min_attributes[j], max_attributes[j])
                tuple_X = tuple_X + (modifiedAttribute)

            else:

                tuple_R = tuple_R + (1)
                tuple_X = tuple_X + (data[j])
            

            tuple_W = tuple_W + (cosine_distance(tuple_X, data))

            X.append(tuple_X)
            R.append(tuple_R)
            W.append(tuple_W)

    Y = [f(x) for x in X]

    G = Ridge(alpha=1.0)
    G.fit(R, Y)

    return 



import numpy as np
import scipy as sc


def LIMEAlgorithm(training_data, perm_number):
    X = []
    R = []
    W = []
    for i in range (1, perm_number):

        atributes = np.random.choice(np.random.randint(1, training_data.size()),np.random.randint(1, training_data.size()))
        for atribute in atributes:
            atributeToModify = training_data[atribute]
            modifiedAtribute = 
            W.append(abs(modifiedAtribute - atributeToModify))
            X.append(modifiedAtribute)



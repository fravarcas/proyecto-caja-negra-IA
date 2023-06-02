import numpy as np
import scipy as sc


def LIMEAlgorithm(f, N):

    X = []
    R = []
    W = []

    for i in range (1, N):

        atributes = np.random.choice(np.random.randint(1, f.size()),np.random.randint(1, f.size()))
        tuple_R = tuple()
        tuple_W = tuple()
        tuple_X = tuple()

        for j in range(0, f.size()):

            if j in atributes:

                tuple_R = tuple_R + (1)
                atributeToModify = f[j]
                modifiedAtribute = 
                tuple_W = tuple_W + (abs(modifiedAtribute - atributeToModify))
                tuple_X = tuple_X + (modifiedAtribute)

            else:

                tuple_R = tuple_R + (0)
                tuple_X = tuple_X + (f[j])
                tuple_W = tuple_W + (0)
            
            X.append(tuple_X)
            R.append(tuple_R)
            W.append(tuple_W)

    Y = [f(x) for x in X]



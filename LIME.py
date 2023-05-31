import numpy as np
import scipy as sc
import lime
from lime import lime_tabular

def LIMEAlgorithm(training_data, column_names, discretize_continuous):

    explaining_alg = lime_tabular.LimeTabularExplainer(training_data, column_names, discretize_continuous)



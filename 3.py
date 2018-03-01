import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import getcwd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
import json

with open(getcwd() + "\\train.json") as f:
    data = pd.DataFrame.from_dict(json.load(f))

cuisine_dictionary = pd.read_csv(filepath_or_buffer=getcwd()+"\\cuisine_dictionary.csv",
                                 index_col=0)

cuisines_types = data['cuisine'].unique()
total_ingr = cuisine_dictionary.sum()

#---------------------------------------
def check_func(ingr):
    best_pred = {"kitchen": "", "score": 0}

    for csn in cuisines_types:
        a = cuisine_dictionary.ix[csn][ingr]
        if best_pred["score"] < a.sum():
            best_pred["score"] = a.sum()
            best_pred["kitchen"] = csn
    return best_pred["kitchen"]

#---------------------------------------
small_data = data.head(1000)

# small_data["csn_pred"] = map(check_func,small_data.ingredients,small_data.index)
small_data["csn_pred"] = [check_func(z) for z in small_data['ingredients']]


# small_data.csn_pred = total_ingr[small_data.ingredients]

# small_data.apply(check_func, axis=1)
print small_data[["csn_pred","cuisine"]]
print classification_report(small_data.cuisine, small_data.csn_pred)
print "Accuracy: ",accuracy_score(small_data.csn_pred, small_data.cuisine)
# cm = confusion_matrix(small_data.cuisine,
#                       small_data.csn_pred,
#                       labels=cuisines_types)
# print pd.DataFrame(cm,
#                    index= cuisines_types,
#                    columns= cuisines_types)

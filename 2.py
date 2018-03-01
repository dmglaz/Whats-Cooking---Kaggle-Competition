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
cuisines_types = data['cuisine'].unique()

cuisine_dictionary = pd.read_csv(filepath_or_buffer=getcwd()+"\\cuisine_dictionary.csv",
                                 index_col=0)

total_ingr = cuisine_dictionary.sum()

small_data = data.head(20)

# print small_data.ix[1]["ingredients"]
# print cuisine_dictionary.ix["greek"][small_data.ix[1]["ingredients"]]
# print (cuisine_dictionary.ix["greek"][small_data.ix[1]["ingredients"]]/\
#         cuisine_dictionary[small_data.ix[1]["ingredients"]].sum()).sum()

# for csn in cuisines_types:
#     small_data[csn + '_score'] = None

small_data["pred"] = total_ingr[small_data["ingredients"]]
small_data.drop("ingredients", 1,  inplace=True)
print small_data
# for csn in cuisines_types:
#     for i in range(20):
#         a = cuisine_dictionary.ix[csn][small_data.ix[i]["ingredients"]]
#         b = total_ingr[small_data.ix[i]["ingredients"]]
#         small_data.loc[i,csn + '_score'] = (a/b).sum()
#         # small_data.ix[i][csn + '_score'] = (a/b).sum()
#         # small_data.(index = i, col = csn + '_score', value=(a/b).sum())
# small_data.drop("ingredients", 1,  inplace=True)
# print small_data


# small_data.drop("ingredients", 1,  inplace=True)
# print small_data

# data['cuisine_pred'] = None
# data['ingr_cnt'] = data['ingredients'].value_counts()
# data.drop("ingredients", 1,  inplace=True)
# print data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import getcwd

from setuptools.command.test import test
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
import json

# get the data
data = pd.read_json(getcwd() + "\\train.json")
test_data = pd.read_json(getcwd() + "\\test.json")
# cuisines_types = data['cuisine'].unique()#array
# cuisines = data.groupby('cuisine')["ingredients"].sum() #DataFrame
# all_ingrerdients = pd.Series(cuisines.sum()).unique() #array
#
# cuisine_dictionary = pd.DataFrame(index=cuisines_types,
#                         columns=all_ingrerdients,
#                         data=np.zeros((len(cuisines_types),len(all_ingrerdients))))
#
# for kitchen in cuisines_types:
#     ingredients_of_kitchen = pd.Series(cuisines[kitchen]).unique()
#     ing_pop_in_cusine = pd.Series(cuisines[kitchen]).value_counts()
#     for ingr in ingredients_of_kitchen:
#         cuisine_dictionary.set_value(kitchen, ingr,
#                                      ing_pop_in_cusine.ix[ingr])
#
# total_ingr = cuisine_dictionary.max()
# cuisine_dictionary = cuisine_dictionary/total_ingr;
# cuisine_dictionary[cuisine_dictionary < 0.5] = 0
# cuisine_dictionary.to_csv(path_or_buf = getcwd() + "\\cuisine_dictionary.csv",
#                           encoding='utf-8-sig')

data['ingr_new'] = [pd.Series(z).str.replace(" ", "_") for z in data['ingredients']]
test['ingr_new'] = [pd.Series(z).str.replace(" ", "_") for z in test['ingredients']]

data.to_csv(path_or_buf = getcwd() + "\\data.csv", encoding='utf-8-sig')
data.to_csv(path_or_buf = getcwd() + "\\test.csv", encoding='utf-8-sig')

cuisines_types = data['cuisine'].unique()
# cuisines = data.groupby('cuisine')["ingr_new"].sum()
all_ingr = pd.Series(data.ingr_new.sum()).unique()

ingr_freq_table_TRAIN = pd.DataFrame(index=data.id,
                                     columns=all_ingr,
                                     data=np.zeros((len(data.id),len(all_ingr))))

ingr_freq_table_TEST = pd.DataFrame(index=test_data.id,
                                    columns=all_ingr,
                                    data=np.zeros((len(test_data.id),len(all_ingr))))

# print data.ix[5].ingredients
# print ingr_freq_table_TRAIN.ix[5][pd.Series(data.ix[5].ingredients)]
# print ingr_freq_table_TEST.ix[2][test_data.ix[2].ingredients]
try:
    for index, val in enumerate(data.id):
        ingr_freq_table_TRAIN.ix[index][data.ix[index].ingr_new] = 1
except:
    print index, val, data.ix[index].ingr_new

print ingr_freq_table_TRAIN
# ingr_freq_table_TRAIN = ingr_freq_table_TRAIN / ingr_freq_table_TRAIN.sum()
# ingr_freq_table_TRAIN['org_cuisine'] = data['cuisine']
# ingr_freq_table_TRAIN.to_csv(path_or_buf = getcwd() + "\\tf_id_matrix_TRAIN.csv",
#                                 encoding='utf-8-sig')
#
# ingr_freq_table_TEST['id'] = test_data.id
# ingr_freq_table_TEST.to_csv(path_or_buf = getcwd() + "\\test_matrix.csv",
#                           encoding='utf-8-sig')

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
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# get the data, prep and split the data-----------------------------------
data = pd.read_json(getcwd() + "\\train.json")
test_data = pd.read_json(getcwd() + "\\test.json")
data['ingr_string'] = [', '.join(pd.Series(z).str.replace(" ", "_")).strip()
                            for z in data['ingredients']]
test_data['ingr_string'] = [', '.join(pd.Series(z).str.replace(" ", "_")).strip()
                                for z in test_data['ingredients']]

cuisines = data.groupby('cuisine')["ingredients"].sum() #DataFrame
train, valid = train_test_split(data, train_size=0.7)

# create a tf_id matrix and create an estimator---------------------------
tf_idf_VECTOR = TfidfVectorizer();
tf_idf_matrix_TRAIN = tf_idf_VECTOR.fit_transform(data.ingr_string)
tf_idf_matrix_TRAIN_DENSE = tf_idf_matrix_TRAIN.todense()

clf = RandomForestClassifier().fit(tf_idf_matrix_TRAIN_DENSE, data['cuisine'])

# #lets see how well the estimator works on the valid set-------------------
tf_idf_matrix_VALID = tf_idf_VECTOR.fit_transform(valid['ingr_string'])
tf_idf_matrix_VALID_DENSE = tf_idf_matrix_VALID.todense()

print tf_idf_matrix_VALID_DENSE
valid["csn_pred"] = clf.predict(tf_idf_matrix_VALID_DENSE)

print classification_report(valid.cuisine, valid.csn_pred)
print "Accuracy: ", accuracy_score(valid.cuisine, valid.csn_pred)

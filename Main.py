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
from Functions import *
from nltk.stem.porter import *

data = get_original_data()

data["train"].columns
data["train"].index

def clean_ing_list(ing_list):
    return [clean_ing(ing) for ing in ing_list]

def clean_ing(ing):
    # remove numbers and chars like (...),[...],{...}, ', "," , . , /, word!, number%,
    # note that im leaving & un touched
    ing = str(ing).lower()
    ing = ing.encode("ascii","ignore") #convert_to_ascii
    ing = re.sub('\((.*)\)\s*', '', ing).strip() #remove anything between ()
    ing = re.sub('\[(.*)\]\s*', '', ing).strip() #remove anything between []
    ing = re.sub('\{(.*)\}\s*', '', ing).strip() #remove anything between {}
    ing = re.sub('\'', '', ing).strip() #remove '
    ing = re.sub('\,', '', ing).strip() #remove ,
    ing = re.sub('\.', '', ing).strip() #remove .
    ing = re.sub('\/', '', ing).strip() #remove /
    ing = re.sub('[a-z]*!\s*', '', ing).strip() #remove ! and any word that comes before it
    ing = re.sub('[a-z0-9]*%\s*', '', ing).strip() #remove % and any number\word that comes before it
    ing = re.sub('-\s*', '', ing).strip() #remove -
    ing = re.sub('[lb|kg|oz]', '', ing).strip() #remove units lb, kg, oz,
    if stopwords.words('english') in ing:



     # in stopwords.words('english') #remove stop words
     # remove all extra white spaces


data["train"]["ingredients_clean"] = [clean_ing_list(ing_list) for ing_list in data["train"].ingredients]


# def ingr_prep(ingr):
#    ingr_1 = re.sub('\((.*)\)\s*', '', ingr).lower().strip()
#    ingr_2 = re.sub(' mix', '', ingr_1).strip()
#    ingr_3 = re.sub('\%', '', ingr_2).strip()
#    ingr_4 = re.sub('[\,\.\']', ' ', ingr_3).strip()
#    ingr_5 = re.sub('.*\/.*to.*lb. ', '', ingr_4).strip()
#    ingr_6 = re.sub('\-', '_', ingr_5).strip()
#    ingr_7 = re.sub('&', '', ingr_6).strip()
#    ingr_8 = re.sub('\'', '', ingr_7).strip()
#    ingr_9 = re.sub('!', '', ingr_8).strip()
#    ingr_10 = ingr_9.encode("ascii","ignore")
#    ingr_LAST = re.sub(' ', '_', ingr_10).strip()
#    return ingr_LAST
#
#
# data['ingr_new'] = [pd.Series([ingr_prep(ingr) for ingr in list]) for list in data['ingredients']]
# data['ingr_string'] = [','.join(z).strip() for z in data['ingr_new']]
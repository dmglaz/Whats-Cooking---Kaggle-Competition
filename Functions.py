import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import getcwd

from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import NaiveBayesClassifier, classify
from random import choice, shuffle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, gutenberg, brown, movie_reviews
from setuptools.command.test import test
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re
import json

def get_original_data():
    data = pd.read_json(getcwd() + "\\original data\\train.json",encoding='utf-8-sig')
    data = data.drop("id", axis=1)
    validate_data = pd.read_json(getcwd() + "\\original data\\test.json",encoding='utf-8-sig')
    return {"train": data, "valid": validate_data}
def get_clean_data():
    data = pd.read_csv(getcwd() + "\\clean data\\train.csv",encoding='utf-8-sig', index_col=0)
    validate_data = pd.read_csv(getcwd() + "\\clean data\\valid.csv",encoding='utf-8-sig', index_col=0)

    return {"train": data, "valid": validate_data}
def clean_and_save_data(data):
    data["train"]["ingredients_clean"] = data["train"]["ingredients"].apply(clean_ing_list)
    data["valid"]["ingredients_clean"] = data["valid"]["ingredients"].apply(clean_ing_list)

    data["train"].to_csv(getcwd() + "\\clean data\\train.csv", sep=',', encoding='utf-8')
    data["valid"].to_csv(getcwd() + "\\clean data\\valid.csv", sep=',', encoding='utf-8')
    return data

def clean_ing_list(ing_list):
    return ",".join([clean_ing(ing) for ing in ing_list])
def clean_ing(ing):
    # remove numbers and chars like (...),[...],{...}, ', "," , . , /, word!, number%,
    # note that im leaving & un touched
    try:
        ing = str(ing).lower()
        # ing = ing.encode("ascii","ignore") #convert_to_ascii
        ing = re.sub('\((.*)\)\s*', '', ing).strip() #remove anything between ()
        ing = re.sub('\[(.*)\]\s*', '', ing).strip() #remove anything between []
        ing = re.sub('\{(.*)\}\s*', '', ing).strip() #remove anything between {}
        ing = re.sub('\'', '', ing).strip() #remove '
        ing = re.sub('\,', '', ing).strip() #remove ,
        ing = re.sub('\.', '', ing).strip() #remove .
        ing = re.sub('\/', '', ing).strip() #remove /
        ing = re.sub('[a-z]*!\s*', '', ing).strip() #remove ! and any word that comes before it
        ing = re.sub('[a-z0-9]*%\s*', '', ing).strip() #remove % and any number\word that comes before it
        ing = re.sub('\s*-\s*', ' ', ing).strip() #remove -

        words_in_ing = ing.split(" ")
        words_in_ing = [good_word for good_word in words_in_ing if not good_word in ["lb","kg", "oz"]]   # in stopwords.words('english')
        ing = " ".join([good_word for good_word in words_in_ing if not good_word in stopwords.words('english')]).strip()   # in stopwords.words('english')
    except Exception as inst:
        print(ing, "Exception")
    return ing
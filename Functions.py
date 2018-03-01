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

def clean_up_data(data):
     data.ingredients
     stopwords.words()

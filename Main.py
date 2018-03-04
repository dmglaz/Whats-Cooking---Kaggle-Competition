import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os import getcwd

from setuptools.command.test import test
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import re, nltk
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from Functions import *
from nltk.stem.porter import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


# data = get_original_data()
# clean_and_save_data(data)
clean_data = get_clean_data()
# ------------------------------------------------------------------------------------
# remove from valid data ingirdients that dont exist in train data
all_train_ingr = set()

for ingr_list in clean_data["train"]["ingredients_clean"]:
    all_train_ingr = all_train_ingr.union(set(ingr_list.split(",")))

valid_clean_ing = []
for ingr_list in clean_data["valid"]["ingredients_clean"]:
    filter_ingr = []
    for ing in ingr_list.split(","):
        if ing in all_train_ingr:
            filter_ingr.append(ing)
    valid_clean_ing.append(",".join(filter_ingr))

clean_data["valid"]["ingredients_clean_filtered"] = pd.Series(valid_clean_ing)

# ------------------------TF_IDF--------------------------------------------------------
vectorizertr = TfidfVectorizer(ngram_range = ( 1 , 1 ),
                               analyzer="word",
                               max_df = .57,
                               binary=False ,
                               token_pattern=r'\w+' ,
                               sublinear_tf=False)
tf_idf = vectorizertr.fit_transform(clean_data["train"]["ingredients_clean"]).todense()
tf_idf_valid = vectorizertr.transform(clean_data["valid"]["ingredients_clean_filtered"]).todense()

# ----------------train test for the classifier from the original train------------------------------

X = {"all": tf_idf}
y = {"all": clean_data["train"]["cuisine"]}
X["train"], X["test"], y["train"], y["test"] = train_test_split(tf_idf,
                                                                clean_data["train"]["cuisine"],
                                                                test_size=0.3)

# ---------------------------------all classifers---------------------------------------------------

classifiers = [
    ('Log_Reg', LogisticRegression()), # train: 0.8 , test: 0.774
    # ('Dec_Tree', DecisionTreeClassifier()), # train: 0.999 , test: 0.616
    # ('Random Forest', RandomForestClassifier()), # train: 0.994 , test: 0.694
    # ('Multinomial NB', MultinomialNB()), # train: 0.68 , test: 0.66
    # # ('SVM', SVC()), #too slow
    ('Liniar SVM', LinearSVC()), # train: 0.864 , test: 0.785
    # ("KNN", KNeighborsClassifier()),  #too slow
    # ("Ada_boost_Log_Reg", AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators = 10, learning_rate=0.01)), #train: 0.19 , test: 0.19
    ("Ada_boost_Liniar_SVM", AdaBoostClassifier(base_estimator=LinearSVC(),n_estimators=15, learning_rate=0.01, algorithm="SAMME")), #train: 0.867 , test: 0.786
    # ("Gradient_boost", GradientBoostingClassifier(max_depth=3,n_estimators=5,learning_rate=0.01)), #too slow
   ]

#find the best classifier
results = pd.DataFrame(index=["train","test"])
for clf_name, clf in classifiers:
    clf.fit(X["train"], y["train"])
    results[clf_name] = [clf.score(X["train"], y["train"]), clf.score(X["test"], y["test"])]
print(results)

# Voting classier
clsf_voting = VotingClassifier(estimators=classifiers, voting="hard")
clsf_voting.fit(X["train"], y["train"])
results["Voting"] = [clsf_voting.score(X["train"], y["train"]), clsf_voting.score(X["test"], y["test"])]
print(results)

# ------------------------------------submission function------------------------------------------------
chosen_clsf = AdaBoostClassifier(base_estimator=LinearSVC(),n_estimators=15, learning_rate=0.01, algorithm="SAMME")
chosen_clsf.fit(X["all"], y["all"])
submision_df = pd.DataFrame(data = chosen_clsf.predict(tf_idf_valid),
                           index = clean_data["valid"].id,
                           columns=["cuisine"])
submision_df.to_csv(getcwd() + "\\Submission\\subm_Ada_boost_Liniar_SVM.csv", sep=',', encoding='utf-8')

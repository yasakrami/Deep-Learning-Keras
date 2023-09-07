
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
st = time.time()
class_number = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = keras.utils.to_categorical(y_train, class_number)
y_test = keras.utils.to_categorical(y_test, class_number)
model = keras.Sequential([keras.Input(shape=input_shape),layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),layers.Flatten(),layers.Dropout(0.5),layers.Dense(class_number, activation="softmax"),])
model.summary()
batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
etM = time.time()
stF =time.time()
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from keras.datasets import fashion_mnist
class_number = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = keras.utils.to_categorical(y_train, class_number)
y_test = keras.utils.to_categorical(y_test, class_number)
model = keras.Sequential([keras.Input(shape=input_shape),layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),layers.Flatten(),layers.Dropout(0.5),layers.Dense(class_number, activation="softmax"),])
model.summary()
batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
etF = time.time()
elapsed_timeM = etM - st
print('execution time of MNIST:' , elapsed_timeM, 'seconds')
elapsed_timeF = stF - etF
print('execution time of fashion MNIST:' , elapsed_timeF, 'seconds')
if elapsed_timeF > elapsed_timeM:
    print('times of fashion Mnist is bigger and this prcess is slower')
else:
    print('times of Mnist is bigger and this prcess is slower')


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
np.random.seed(0)
def load_data():
    data_loc = "spambase.data"
    cols_loc = "spambase.names"
    col_names = []
    with open(cols_loc, "r") as f:
        i = 0
        for line in f:
            i += 1
            if i < 34:
                continue
            ls = line.rstrip().split(":")
            col_names.append(ls[0])
    col_names.append("spam")
    data = []
    with open(data_loc, "r") as f:
        for line in f:
            ls = list(map(np.float32, line.rstrip().split(",")))
            data.append(ls)
    minmax = MinMaxScaler()
    data_normalized = minmax.fit_transform(data)
    return pd.DataFrame(data_normalized, columns=col_names)
def hyperparameter_search(clf, params, X, y):
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True)
    try:
        model_search = RandomizedSearchCV(clf, params, n_iter=10, cv=cross_validation)
        model_search.fit(X, y)
        return model_search.best_estimator_
    except:
        return clf
df = load_data()
print(df.head())
print()
corr = df.corr()
print(corr["spam"].sort_values(ascending=False))
print()
X = df.drop("spam", axis=1)
y = df["spam"]
X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
classifier_names = ["GaussianNB", "MultinomialNB", "BernoulliNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"]
classifiers = {"GaussianNB": GaussianNB(),"MultinomialNB": MultinomialNB(),"BernoulliNB": BernoulliNB(),"LogisticRegression": LogisticRegression(),"DecisionTreeClassifier": DecisionTreeClassifier(),"RandomForestClassifier": RandomForestClassifier(), "XGBClassifier": XGBClassifier()}
params = {"GaussianNB": {}, "MultinomialNB": {"alpha": np.random.rand(1000)},"BernoulliNB": {"alpha": np.random.rand(100),"binarize": np.random.rand(300)},"LogisticRegression": {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]},"DecisionTreeClassifier": {"criterion": ["gini", "entropy"],"splitter": ["best", "random"],"max_features": [None, "auto", "sqrt", "log2"]},"RandomForestClassifier": {"n_estimators": [8, 10, 13, 15, 20, 24, 28],"criterion": ["gini", "entropy"],"max_depth": [None, 2, 3, 5, 7], "max_features": [None, "auto", "sqrt", "log2"]}, "XGBClassifier": {"max_depth": [2, 3, 5, 6, 7, 9, 10], "n_estimators": [50, 100, 125, 140, 175, 200]}}
for clf_name in classifier_names:
    clf = classifiers[clf_name]
    par = params[clf_name]
    best_clf = hyperparameter_search(clf, par, X_train, y_train)
    print(best_clf)
    best_clf.fit(X_train, y_train)
    print("Score {0:.9f}%".format(100. * best_clf.score(X_test, y_test)))
    print()


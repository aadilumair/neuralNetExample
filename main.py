import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("data.csv") #load data

train, test = train_test_split(data, random_state=42)#split data into test and train
X_train = train[train.columns[2:31]]
Y_train = train['diagnosis']
X_test = test[train.columns[2:31]]
Y_test = test['diagnosis']

scalar = preprocessing.StandardScaler() #converting data to scalar format
scalar.fit(X_train)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

MLP = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, activation='logistic') #setting up Neural Network
MLP.fit(X_train, Y_train.values.ravel())

predictions = MLP.predict(X_test)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
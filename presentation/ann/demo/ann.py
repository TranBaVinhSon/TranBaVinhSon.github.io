import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import keras 
from keras.models import Sequential
from keras.layers import Dropout, Dense
### PART 1: Prepare Data
# Importing the dataset 
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()

#remove first column
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
y_test = sc.transform(X_test)


### PART 2: Build ANN 
# Two way to define ANN: Define the graph or sequence
classifier = Sequential()
# First Hidden Layer
# Number of Node = Number of Independent Variable (input_dim = 11)
# Uniform function to initializer weight
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
classifier.add(Dropout(rate=0.1))
# Second Hidden Layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(rate=0.1))
# Output Layer 
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compiling the ANN 
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
# Fitting the ANN to the Training set 
classifier.fit(X_train, y_train, batch_size=10, epochs=10)

### Parameters Tuning

### PART 3: Making the predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


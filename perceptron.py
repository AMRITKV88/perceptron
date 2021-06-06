# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Load Dataset

# %%
from sklearn import datasets
import numpy as np
import pandas as pd
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target


# %%
print(X)
print(breast_cancer.feature_names)
print(X.shape)

# %% [markdown]
# Dataset preparation

# %%
# Dataset preparation
data_with_feature_name = pd.DataFrame(data = X, columns = breast_cancer.feature_names)
data = data_with_feature_name
data['class'] = Y

# %% [markdown]
# Train and Test Split

# %%
# Train and Test Split
from sklearn.model_selection import train_test_split
X_data = data.drop('class', axis=1)
Y_data = data['class']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, stratify = Y_data, random_state = 1)

# %% [markdown]
# Perceptron Class

# %%
from sklearn.metrics import accuracy_score
X_train = X_train.values
X_test = X_test.values


# %%
class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):

        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        y = []
        for x in X:
            y_pred = self.model(x)
            y.append(y_pred)
        return np.array(y)

    def fit(self, X, Y, epochs = 10, lr = 0.1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        w_matrix = []
        accuracy = {}
        max_accuracy = 0

        for i in range(epochs):
            for x,y in zip(X, Y):
                y_train_pred = self.model(x)
                if y_train_pred == 1 and y == 0:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1
                elif y_train_pred == 0 and y == 1:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
            w_matrix.append(self.w)
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                check_point_w = self.w
                check_point_b = self.b

        self.w = check_point_w
        self.b = check_point_b

        print("Maximum training accuracy :",max_accuracy)
        return np.array(w_matrix)


# %%
perceptron = Perceptron()
weight_mtx = perceptron.fit(X_train, Y_train, 1000, 0.1)
y_pred_test = perceptron.predict(X_test)
print("Testing accuracy : ", accuracy_score(y_pred_test, Y_test))


# %%



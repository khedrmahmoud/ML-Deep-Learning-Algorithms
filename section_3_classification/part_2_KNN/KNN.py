"""
@author: Khedr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %% Spliting data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# %% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# %% Fitting the Classification Model
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
classifier.fit(x_train, y_train)
# %% Predicting the test set
y_pred = classifier.predict(x_test)

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# %% Visualizing the Trainging set result
from matplotlib.colors import ListedColormap


def visualize(x_set, y_set, title, classifier):
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))  # Added classifier parameter and fixed cmap

    plt.xlim(x1.min(), x1.max())  # Fixed min and max functions
    plt.ylim(x2.min(), x2.max())  # Fixed min and max functions

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(
            ('red', 'green'))(i), label=j)  # Fixed color mapping

    plt.title('K-NN ('+title+')')
    plt.xlabel("age")
    plt.ylabel("Estimated salary")
    plt.legend()
    plt.show()


visualize(x_train, y_train, "Training", classifier)
visualize(x_test, y_test, "test", classifier) 
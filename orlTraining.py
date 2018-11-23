from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
# Import the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd
import time

# Construct the colormap
current_palette = sns.color_palette("muted", n_colors=40)
cmap = ListedColormap(sns.color_palette(current_palette).as_hex())

from NearestCentroid import NearestCentroid
from NearestSubclassCentroid import NearestSubclassCentroid
from ORL.loadORL import loadORL
from Perceptron_multiclass_backpropagation import get_classifier
from Perceptron_multiclass_lms import Perceptron_multiclass_lms
from kNearestNeighbor import kNearestNeighbor

# load in data
filesToLoad = {'images': 'ORL/orl_data.mat', 'labels': 'ORL/orl_lbls.mat'}
x_train, x_test, y_train, y_test = loadORL(filesToLoad)

scaler = StandardScaler()

# Standardizing the features
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# Make an instance of the Model
pca = PCA(n_components=2)

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# Create a scatter plot of the data
plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test, cmap=cmap)
plt.title("Scatterplot ORLData")

# Add a color bar
plt.colorbar()

# Show the plot
plt.show()
plt.savefig("scatter_plot_Orl.png", bbox_inches='tight')

# making our predictions
predictions = []

# Accuracy data:
accuracy_table_data = []

# KNearestNeighbor
kNearestNeighbor(x_train, y_train, x_test, predictions, 3)
# evaluating accuracy
accuracy_knn = accuracy_score(y_test, predictions)
accuracy_table_data.append(["KNearestNeighbor", accuracy_knn])

# Perceptron Multiclass Least Mean Square
perc_lms = Perceptron_multiclass_lms();
perc_lms.train(x_train, y_train)
predictions = perc_lms.predict(x_test)
accuracy_lms = accuracy_score(y_test, predictions)
accuracy_table_data.append(["Perceptron Multiclass Least Mean Square", accuracy_lms])

# Perceptron Multiclass Backpropagation:
Perceptron_backpropagation = get_classifier();
Perceptron_backpropagation.fit(x_train, y_train)
predictions = Perceptron_backpropagation.predict(x_test)
accuracy_backpropagation = accuracy_score(y_test, predictions)
accuracy_table_data.append(["Perceptron Backpropagation", accuracy_backpropagation])

# NearestCentroid
# Normal data
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
predictions = ncc.predict(x_test)
accuracy_ncc = accuracy_score(y_test, predictions)
accuracy_table_data.append(["NearestCentroid", accuracy_ncc])

# Nearest subclass classifier (2 subclasses):
nsc = NearestSubclassCentroid()
nsc.fit(x_train, y_train, 2)
predictions = nsc.predict(x_test)
accuracy_nsc2 = accuracy_score(y_test, predictions)
accuracy_table_data.append(["Nearest subclass classifier 2", accuracy_nsc2])

# Nearest subclass classifier (3 subclasses):
nsc3 = NearestSubclassCentroid()
nsc3.fit(x_train, y_train, 3)
predictions = nsc3.predict(x_test)
accuracy_nsc3 = accuracy_score(y_test, predictions)
accuracy_table_data.append(["Nearest subclass classifier 3", accuracy_nsc3])

# Nearest subclass classifier (5 subclasses):
nsc5 = NearestSubclassCentroid()
nsc5.fit(x_train, y_train, 5)
predictions = nsc5.predict(x_test)
accuracy_nsc5 = accuracy_score(y_test, predictions)
accuracy_table_data.append(["Nearest subclass classifier 5", accuracy_nsc5])


#2 Data
accuracy2d = []
predictions = []
# KNearestNeighbor
kNearestNeighbor(x_train_pca, y_train, x_test_pca, predictions, 3)
# evaluating accuracy
accuracy_knn = accuracy_score(y_test, predictions)
accuracy2d.append(["KNearestNeighbor", accuracy_knn])

# Perceptron Multiclass Least Mean Square
perc_lms = Perceptron_multiclass_lms();
perc_lms.train(x_train_pca, y_train)
predictions = perc_lms.predict(x_test_pca)
accuracy_lms = accuracy_score(y_test, predictions)
accuracy2d.append(["Perceptron Multiclass Least Mean Square", accuracy_lms])

# Perceptron Multiclass Backpropagation:
Perceptron_backpropagation = get_classifier();
Perceptron_backpropagation.fit(x_train_pca, y_train)
predictions = Perceptron_backpropagation.predict(x_test_pca)
accuracy_backpropagation = accuracy_score(y_test, predictions)
accuracy2d.append(["Perceptron Backpropagation", accuracy_backpropagation])

# NearestCentroid
# Normal data
ncc = NearestCentroid()
ncc.fit(x_train_pca, y_train)
# get the model accuracy
predictions = ncc.predict(x_test_pca)
accuracy_ncc = accuracy_score(y_test, predictions)
accuracy2d.append(["NearestCentroid", accuracy_ncc])

# Nearest subclass classifier (2 subclasses):
nsc = NearestSubclassCentroid()
nsc.fit(x_train_pca, y_train, 2)
predictions = nsc.predict(x_test_pca)
accuracy_nsc2 = accuracy_score(y_test, predictions)
accuracy2d.append(["Nearest subclass classifier 2", accuracy_nsc2])

# Nearest subclass classifier (3 subclasses):
nsc3 = NearestSubclassCentroid()
nsc3.fit(x_train_pca, y_train, 3)
predictions = nsc3.predict(x_test_pca)
accuracy_nsc3 = accuracy_score(y_test, predictions)
accuracy2d.append(["Nearest subclass classifier 3", accuracy_nsc3])

# Nearest subclass classifier (5 subclasses):
nsc5 = NearestSubclassCentroid()
nsc5.fit(x_train, y_train, 5)
predictions = nsc5.predict(x_test)
accuracy_nsc5 = accuracy_score(y_test, predictions)
accuracy2d.append(["Nearest subclass classifier 5", accuracy_nsc5])

df = pd.DataFrame()
df['Algortihms'] = [e[0] for e in accuracy_table_data]
df['Accuracy full data set'] = [e[1] for e in accuracy_table_data]
df['Accuracy pca data (2D)'] = [e[1] for e in accuracy2d]
with open('Orltable.tex','w') as tf:
    tf.write(df.to_latex())

print(df.all)
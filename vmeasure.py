# importing pandas module
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

# making data frame
df = pd.read_csv('creditcard.csv')

# Separating the dependent and independent variables
y = df['Class']
X = df.drop('Class', axis=1)

# display
print(X)

# List of V-Measure Scores for different models
v_scores = []

# List of different types of covariance parameters
N_Clusters = [2, 3, 4, 5, 6]

# Building the clustering model
kmeans2 = KMeans(n_clusters=2)

# Training the clustering model
kmeans2.fit(X)

# Storing the predicted Clustering labels
labels2 = kmeans2.predict(X)

# Evaluating the performance
v_scores.append(v_measure_score(y, labels2))

# Building the clustering model
kmeans3 = KMeans(n_clusters=3)

# Training the clustering model
kmeans3.fit(X)

# Storing the predicted Clustering labels
labels3 = kmeans3.predict(X)

# Evaluating the performance
v_scores.append(v_measure_score(y, labels3))

# Building the clustering model
kmeans4 = KMeans(n_clusters=4)

# Training the clustering model
kmeans4.fit(X)

# Storing the predicted Clustering labels
labels4 = kmeans4.predict(X)

# Evaluating the performance
v_scores.append(v_measure_score(y, labels4))


# Building the clustering model
kmeans5 = KMeans(n_clusters=5)

# Training the clustering model
kmeans5.fit(X)

# Storing the predicted Clustering labels
labels5 = kmeans5.predict(X)

# Evaluating the performance
v_scores.append(v_measure_score(y, labels5))


# Building the clustering model
kmeans6 = KMeans(n_clusters=6)

# Training the clustering model
kmeans6.fit(X)

# Storing the predicted Clustering labels
labels6 = kmeans6.predict(X)

# Evaluating the performance
v_scores.append(v_measure_score(y, labels6))


# Plotting a Bar Graph to compare the models
plt.bar(N_Clusters, v_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('V-Measure Score')
plt.title('Comparison of different Clustering Models')
plt.show()

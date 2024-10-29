import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv("/cities_r2.csv")

data.head(5)

data.describe()

data.info()

data.isna().sum()

columns_for_clustering = ['total_graduates']

# Extracting the relevant columns
selected_data = data[columns_for_clustering]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

scaled_data = sc.fit_transform(selected_data)

cluster_score=[]

for i in range(1,25):
    kmeans = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans.fit(scaled_data)
    cluster_score.append(kmeans.inertia_)


plt.figure(figsize=(10,4))
plt.plot(range(1,25), cluster_score, color="blue", linestyle="dashed", marker='o', markerfacecolor='red', markersize=10)
plt.title("Finding number of clusters using ELBOW method")
plt.xlabel('No of clusters')
plt.ylabel('Clustering score')


kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(scaled_data)

pred = kmeans.predict(scaled_data)

data['Cluster'] = pd.DataFrame(pred, columns=['Cluster'])

print('Number of data points in each cluster=\n', data['Cluster'].value_counts())


kmeans.cluster_centers_

# Visualizing the clusters
# Plot in 1D for a single feature
plt.figure(figsize=(10, 6))
plt.scatter(data['total_graduates'], [0]*len(data), c=data['Cluster'], marker='o', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], [0]3, marker='', s=200, color="red")  # Center points for 1D clustering
plt.title('K-Means Clustering on Total Graduates')
plt.xlabel('Total Graduates')
plt.show()
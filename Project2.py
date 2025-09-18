import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('rolling_stones_spotify.csv')

df = df.drop_duplicates()

albums = df['album'].drop_duplicates()

print(albums)

num_albums = df['album'].value_counts()

plt.bar(albums, num_albums, color='blue', edgecolor='black', label='Bar chart')
plt.show()

features = df[['popularity','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence', 'duration_ms']]

sns.pairplot(features)
plt.show()

correlation_matrix = features.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

scaler = StandardScaler()
features_Scaled = scaler.fit_transform(features)

print(features_Scaled)

features = features.values

wcss = []
for i in range(1, 11):
  model = KMeans(n_clusters = i, n_init=10, init= 'k-means++', random_state=0)
  model.fit(features)
  wcss.append(model.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel("Number of clusters")
plt.ylabel('WCSS')
plt.show()

model = KMeans(n_clusters = 5, n_init=10, init='k-means++', random_state=0)
y_kmeans = model.fit_predict(features)


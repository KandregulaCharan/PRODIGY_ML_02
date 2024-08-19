import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/KandregulaCharan/PRODIGY_ML_02/main/mall_customer.py')

print("Data Head:")
print(df.head())

features = df[['total_spent', 'num_transactions', 'avg_transaction_value', 'purchase_frequency']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

silhouette_scores = []
for k in k_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(k_range[1:], silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)
plt.show()


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

print("Clustered Data:")
print(df.head())

cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

#Visualize clusters if you have 2D features
plt.figure(figsize=(10, 6))
plt.scatter(df['total_spent'], df['num_transactions'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Total Spent')
plt.ylabel('Number of Transactions')
plt.title('Customer Clusters')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

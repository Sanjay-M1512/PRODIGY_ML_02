import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Intern\Mall_Customers.csv")

print("First few rows of the dataset:")
print(df.head())


df.drop(columns=['CustomerID'], inplace=True)

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']])

df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender'])

wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method to Find Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(df_scaled)

df['Cluster'] = kmeans.labels_

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df, s=100)
plt.title('Customer Segments Based on Purchase History')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

sil_score = silhouette_score(df_scaled, kmeans.labels_)
print(f"Silhouette Score: {sil_score:.2f}")

cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary (Average Features per Cluster):")
print(cluster_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=df, s=100)
plt.title('Annual Income vs Spending Score for Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', hue='Gender', data=df)
plt.title('Gender Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Membaca data dari file CSV
data = pd.read_csv('mc.csv')

# Memilih hanya kolom numerik
data_numeric = data.select_dtypes(include=[float, int])

# Melihat data numerik
print("Data Numerik:")
print(data_numeric.head())

# Normalisasi Z-score
scaler_zscore = StandardScaler()
data_scaled_zscore = scaler_zscore.fit_transform(data_numeric)
data_normalized_zscore = pd.DataFrame(data_scaled_zscore, columns=data_numeric.columns)

# Normalisasi Min-Max
scaler_minmax = MinMaxScaler()
data_scaled_minmax = scaler_minmax.fit_transform(data_numeric)
data_normalized_minmax = pd.DataFrame(data_scaled_minmax, columns=data_numeric.columns)

# Melihat data setelah normalisasi
print("\nData Setelah Normalisasi Z-score:")
print(data_normalized_zscore.head())
print("\nData Setelah Normalisasi Min-Max:")
print(data_normalized_minmax.head())

# Metode Ward untuk klasterisasi
Z_zscore = linkage(data_normalized_zscore, method='ward')
Z_minmax = linkage(data_normalized_minmax, method='ward')
Z_original = linkage(data_numeric, method='ward')

# Looping untuk jumlah klaster dari 2 hingga 11 dan menghitung Davies-Bouldin Index
dbi_scores_zscore = {}
dbi_scores_minmax = {}
dbi_scores_original = {}

for max_clusters in range(2, 12):
    # Klasterisasi dengan data dinormalisasi (Z-score)
    clusters_zscore = fcluster(Z_zscore, max_clusters, criterion='maxclust')
    dbi_zscore = davies_bouldin_score(data_normalized_zscore, clusters_zscore)
    dbi_scores_zscore[max_clusters] = dbi_zscore

    # Klasterisasi dengan data dinormalisasi (Min-Max)
    clusters_minmax = fcluster(Z_minmax, max_clusters, criterion='maxclust')
    dbi_minmax = davies_bouldin_score(data_normalized_minmax, clusters_minmax)
    dbi_scores_minmax[max_clusters] = dbi_minmax

    # Klasterisasi dengan data asli
    clusters_original = fcluster(Z_original, max_clusters, criterion='maxclust')
    dbi_original = davies_bouldin_score(data_numeric, clusters_original)
    dbi_scores_original[max_clusters] = dbi_original

    # Menyimpan hasil klasterisasi dengan data dinormalisasi (Z-score) ke file CSV
    data['Cluster_Zscore'] = clusters_zscore
    output_filename_zscore = f'cobaml_clustered_zscore_{max_clusters}_clusters.csv'
    data.to_csv(output_filename_zscore, index=False)

    # Menyimpan hasil klasterisasi dengan data dinormalisasi (Min-Max) ke file CSV
    data['Cluster_MinMax'] = clusters_minmax
    output_filename_minmax = f'cobaml_clustered_minmax_{max_clusters}_clusters.csv'
    data.to_csv(output_filename_minmax, index=False)

    # Menyimpan hasil klasterisasi dengan data asli ke file CSV
    data['Cluster_Original'] = clusters_original
    output_filename_original = f'cobaml_clustered_original_{max_clusters}_clusters.csv'
    data.to_csv(output_filename_original, index=False)

    # Menampilkan hasil untuk setiap jumlah klaster
    print(f"\nJumlah klaster: {max_clusters}, Nilai DBI (Z-score): {dbi_zscore}, Nilai DBI (Min-Max): {dbi_minmax}, Nilai DBI (Asli): {dbi_original}")

# Menentukan jumlah klaster dengan nilai DBI terbaik untuk setiap metode
best_num_clusters_zscore = min(dbi_scores_zscore, key=dbi_scores_zscore.get)
best_dbi_zscore = dbi_scores_zscore[best_num_clusters_zscore]

best_num_clusters_minmax = min(dbi_scores_minmax, key=dbi_scores_minmax.get)
best_dbi_minmax = dbi_scores_minmax[best_num_clusters_minmax]

best_num_clusters_original = min(dbi_scores_original, key=dbi_scores_original.get)
best_dbi_original = dbi_scores_original[best_num_clusters_original]

print(f"\nJumlah klaster terbaik (Z-score): {best_num_clusters_zscore}, Nilai DBI: {best_dbi_zscore}")
print(f"Jumlah klaster terbaik (Min-Max): {best_num_clusters_minmax}, Nilai DBI: {best_dbi_minmax}")
print(f"Jumlah klaster terbaik (Asli): {best_num_clusters_original}, Nilai DBI: {best_dbi_original}")

# Plot 3D untuk visualisasi klaster menggunakan jumlah klaster terbaik dari data dinormalisasi (Z-score)
clusters_zscore = fcluster(Z_zscore, best_num_clusters_zscore, criterion='maxclust')
data['Cluster_Zscore'] = clusters_zscore

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_normalized_zscore.iloc[:, 0], data_normalized_zscore.iloc[:, 1], data_normalized_zscore.iloc[:, 2],
           c=clusters_zscore, cmap='viridis', marker='o')
ax.set_title(f'3D Cluster Plot (Best Clusters Z-score: {best_num_clusters_zscore})')
ax.set_xlabel(data_numeric.columns[0])
ax.set_ylabel(data_numeric.columns[1])
ax.set_zlabel(data_numeric.columns[2])
plt.show()

# Plot 3D untuk visualisasi klaster menggunakan jumlah klaster terbaik dari data dinormalisasi (Min-Max)
clusters_minmax = fcluster(Z_minmax, best_num_clusters_minmax, criterion='maxclust')
data['Cluster_MinMax'] = clusters_minmax

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_normalized_minmax.iloc[:, 0], data_normalized_minmax.iloc[:, 1], data_normalized_minmax.iloc[:, 2],
           c=clusters_minmax, cmap='viridis', marker='o')
ax.set_title(f'3D Cluster Plot (Best Clusters Min-Max: {best_num_clusters_minmax})')
ax.set_xlabel(data_numeric.columns[0])
ax.set_ylabel(data_numeric.columns[1])
ax.set_zlabel(data_numeric.columns[2])
plt.show()

# Plot 3D untuk visualisasi klaster menggunakan jumlah klaster terbaik dari data asli
clusters_original = fcluster(Z_original, best_num_clusters_original, criterion='maxclust')
data['Cluster_Original'] = clusters_original

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_numeric.iloc[:, 0], data_numeric.iloc[:, 1], data_numeric.iloc[:, 2],
           c=clusters_original, cmap='viridis', marker='o')
ax.set_title(f'3D Cluster Plot (Best Clusters Original: {best_num_clusters_original})')
ax.set_xlabel(data_numeric.columns[0])
ax.set_ylabel(data_numeric.columns[1])
ax.set_zlabel(data_numeric.columns[2])
plt.show()

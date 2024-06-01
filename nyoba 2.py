import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Membaca data dari file CSV
data = pd.read_csv('cobaml.csv')

# Memilih hanya kolom numerik
data_numeric = data.select_dtypes(include=[float, int])

# Melihat data numerik
print("Data Numerik:")
print(data_numeric.head())

# Metode Ward untuk klasterisasi tanpa normalisasi
Z = linkage(data_numeric, method='ward')

# Membuat dendrogram untuk visualisasi klaster
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram Ward')
plt.xlabel('Data Points')
plt.ylabel('Euclidean distances')
plt.show()

# Memotong dendrogram untuk mendapatkan klaster (misalnya, 3 klaster)
max_clusters = 3
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# Menambahkan kolom klaster ke data asli
data['Cluster'] = clusters

# Melihat data dengan klaster
print("\nData dengan Klaster:")
print(data.head())

# Menyimpan hasil ke file CSV baru
data.to_csv('cobaml_clustered.csv', index=False)

print("\nHasil klaster telah disimpan ke 'cobaml_clustered.csv'")

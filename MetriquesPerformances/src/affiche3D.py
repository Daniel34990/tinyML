import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Chargement des données
data_path = 'tableauSIG.csv'
data = pd.read_csv(data_path)


data['Accuracy^2/Time ratio'] = (data[' Accuracy'].astype(float) ** 2) / data['Training Time (s)']

# Création du graphique 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Axes et données
x = data['Hidden Layer']
y = data['Neuron per hidden layer']
z = data['Accuracy^2/Time ratio']  # Utilisation de la colonne 'Accuracy'

# Tracé du graphique
sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

# Etiquettes
ax.set_xlabel('Nombre de Couches Cachées')
ax.set_ylabel('Nombre de Neurones par Couche')
ax.set_zlabel('Accuracy^2/Time ratio')

# Colorbar
plt.colorbar(sc, ax=ax, label='Accuracy^2/Time ratio')

# Affichage
plt.show()

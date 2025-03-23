import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Chemin vers le fichier CSV
file_path = 'confusion.csv'

# Lire les données
data = pd.read_csv(file_path)

# Assurez-vous que les noms de colonnes 'vrai_label' et 'pred_label' correspondent à votre fichier CSV
mat_conf = confusion_matrix( data['pred_label'],data['vrai_label'])

# Afficher la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(mat_conf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Vrais')
plt.ylabel('Prédits')
plt.title('Matrice de confusion avec 2 couches cachées et 9 neurones pour chacune')
plt.show()
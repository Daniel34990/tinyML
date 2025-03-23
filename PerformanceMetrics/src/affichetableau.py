import pandas as pd
from tkinter import *
from pandastable import Table

# Chargement et préparation des données
data_path = 'tableauSIG.csv'
data = pd.read_csv(data_path)
data_sorted = data.sort_values(by=' Accuracy', ascending=False)
data_sorted['3 Meilleures Précisions'] = ['3 Meilleures précisions en MNIST ANN'] * 3 + [''] * (len(data_sorted) - 3)

# Ajouter une nouvelle colonne pour la métrique Accuracy^2 / Time
#data['Accuracy^2/Time'] = (data[' Accuracy'] ** 2) / data['Training Time (s)']

# Trier les données par la nouvelle métrique en ordre décroissant
#data_sorted = data.sort_values(by='Accuracy^2/Time', ascending=False)

#data_sorted['3 Meilleures Précisions] = ['3 Meilleures Précisions en MNIST ANN'] * 3 + [''] * (len(data_sorted) - 3)

top_three_with_all_columns = data_sorted.head(3)

# Création de la fenêtre
root = Tk()
root.title("Top 3 Meilleures Précisions")

# Création du frame qui contiendra la table
frame = Frame(root)
frame.pack(fill='both', expand=True)

# Ajout de la table à la fenêtre
pt = Table(frame, dataframe=top_three_with_all_columns, showtoolbar=True, showstatusbar=True)
pt.show()

# Exécuter la fenêtre
root.mainloop()
import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV
data = pd.read_csv('training_curve_1hiddenlayers_8hidden.csv')
f=data['validation_error'].max()/data['cross_entropy'].max()

# Tracer l'entropie croisée
plt.figure(figsize=(10, 5))
plt.plot(data['generation'], f*data['cross_entropy'], label='cross_entropy')

# Tracer l'erreur de validation
plt.plot(data['generation'], data['validation_error'], label='Validation Error', linestyle='--')

plt.xlabel('Génération')
plt.ylabel('Valeur')
plt.title('Courbe de Validation')
plt.legend()
plt.show()
mer. 28/02: Rédaction du rapport initial et de la présentation

mer. 13/3:
(Tout le monde) Visonnage des 4 Vidéos de 3BlueOneBrown pour une introduction au deep learning et machine learning (recommandées par l'encadrant).

mer. 20/03:
(Tout le monde)Lecture du livre DeepLearning de Ian Goodfellow.

mar. 26/03:
(Tout le monde): Lecture et Compréhension de la librairie Gennann. Premiers test sur MNIST.

mar. 2/04:
Daniel:
- Ajout de l'entropie croisée comme métrique pour l'early stopping: cela ne change en rien le résultat final.
- Mise en place des tests sur RasbperryPi
Roann:
- Ajout des matrices, et séparation du code pour lire les données
- Ajout des en-têtes pour les types Sequentials & Pooling 
Antoine:
- Mise à jour du dockerfile

mar. 9/04:
Antoine:
- Création d'une branche afin d'implémenter d'un bout de programme fonctionnant en multithreading pour mesurer la quantité de mémoire globale utilisée pendant l'exécution du programme -> critiques vis-à-vis de l'utilisation de celle-ci, elle peut être utilisée par des programmes externes (OS)
Daniel: 
- Ajout d'un fichier pooling.c qui implémente pooling_init et pooling_run.
Roann:
- Ajout des couches de convolutions
Edouard:
- Test des Dense sur RaspberryPi

mer. 24/04:
(Roann, Daniel) Ajout de la structure de couche de convolution 
(Antoine, Edouard) Travail sur la quantité de mémoire utilisée, on cherche à mesurer la mémoire utilisée par le process

mer. 01/05:
(Tout le monde): Rédaction du rapport de mi-parcours et de la présentation mi-parcours

mar. 07/05:
Daniel: 
- Correction de convolution_train et ajout de sequential_train.
Roann:
- Correction du module sequential
- Ajout de la sauvegarde / chargement de modèle
Antoine:
- Correction module gestion des performances en changeant la manière dont sont gérés les FILE
Edouard:
- Connexion RaspberryPi


mer. 15/05 - ven. 17/05:
(Tout le monde): Rédaction de la fiche: Impacts sociaux et environnementaux de votre projet

mer. 22/05:
Roann:
- Mise à jour de `SUIVI.md`
- Ajout de documentation supplémentaire
- Correction d'avertissements
Daniel:
- Correction du code
Antoine:
- Création d'une fonction spécifique pour la gestion des données des performances
Edouard:
- Optimisation des container docker

mer. 29/05:
(tout le monde): Audit du projet

mer. 05/06:
Antoine: 
- Correction du fichier dockerfile sur la branche performance_systeme (elle différait de celle du git suite aux différents tests menés) 
- Modification du main afin d'obtenir les informations de performance du process en question, et non pas les informations globales
Roann:
- Activation: Utilisation des tenseurs au lieu de pointeurs directs
- Convolutions: Support de plusieurs filtres
- MNIST: Ajout de "batch"
Edouard:
- Version 1 du poster
Daniel:
- Correction d'une du code de la rétropropagation dans les CNN

mer. 12/06:
(tout le monde): rendez-vous avec Leonardo pour faire le point sur l'avancement du projet et définir les exigences attendues en vue de la semaine finale.

lun. 24/06:
Daniel:
- Mise en place d'une démo compatible avec les CNN
- Etude de la base de données EMINIST qui inclut des lettres
Roann:
- Traduction de la demo en Web-Assembly
Antoine:
- Configuration d'un serveur Web pour la réception des images de la demo
Edouard:
- Réalisation d'un serveur prédisant la classification d'une image avec notre modèle de TinyML, en écoutant via un socket.

mar. 25/06:
Roann:
- MnistDb: Ajout d'un moyen de bruiter les données
- Mise en place de la connexion entre le serveur Web et le serveur IA qui traite les donées pour l'entraînement et la prédiction du modèle
Antoine:
- Codage de la page web qui contiendra la demo
Daniel:
- Ecriture et entraînement du modèle qui inclut les lettres EMNIST
Edouard:
- Amélioration des serveurs IA et Web 

mer. 26/06
Roann:
- Finalisation de server.c
Antoine:
- Mise en place du lien client / serveur / raspberry
Daniel:
- Entrainement du modèle et tests de ses performances
Edouard:
- Mise en place des dockers web et ML avec docker compose

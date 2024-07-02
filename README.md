Membres du projet:
CANTEL Roann, AKBARINIA Daniel, POMPÉE Edouard, HEITZMANN Antoine

**Présentation :**
Le projet "Apprentissage automatique avec TinyML" vise à implémenter des algorithmes de machine learning, notamment de deep learning (réseaux de neurones), sur des microcontrôleurs à faible consommation énergétique. 
Nous développons nos modèles en langage C pour optimiser l'efficacité, la taille, la consommation, le temps d’entraînement et d'exécution. 
En outre, le tinyML soulève différents enjeux:
- Les enjeux sociaux incluent la confidentialité des données en traitant localement les informations et l'amélioration de l'accessibilité technologique pour les communautés marginalisées. 
- Les enjeux environnementaux concernent la réduction de la consommation énergétique et l'application de TinyML dans des solutions écologiques et durables, comme la gestion optimisée de l'irrigation agricole.



**Explication de l'architecture du git :**
- _Docker_ contient 4 sous-dossiers fonctionnant de manière indépendante grâce à des dockers.
    - _Demo finale_ est le dossier principal qui contient les docker mettant en place le site web de démonstration. Comme expliqué dans le rapport final, l'architecture adoptée est une architecture multi-ports.
    - _IRIS_container_ contient le docker de la première version de tests. Ce programme provient directement du git de gennan et sert de vérification pour la bonne installation de la librairie.
    - _MNIST_container_train_, comme son nom l'indique, ce docker implémente l'entrainement avec les données de la base de données MNIST.
    - _MNIST_run_ contient le docker qui s'occupe de lancer des prévisions à partir d'un réseau de neurones pré-entrainé.
- _MetriquesPerformances_ rassemble les fichiers permettant de mesurer les performances du réseau. Il contient également les fichiers pythons permettant d'afficher les graphes montrés dans le rapport final.
- _Rendus_ rassemble l'ensemble des fichiers demandés et qui composent notre projets :
    - Les différents rapports (initial, intermédiaire et final)
    - Le poster
    - Les impacts sociaux et environnementaux du projet
    - _demo_finale_ qui contient les mêmes informations que _Demo finale_ dans le dossier _Docker_, il permet de mettre en place le site web de démonstration
- _mnist_data_ contient tout simplement les données de la base de données MNIST, ainsi que les fichiers permettant de la librairie
- _server_ contient le serveur accessible pour la partie client et met en place la page web. Il peut être supprimé car redondant avec _demo_finale_
- _src_ contient l'ensemble des fichiers permettant le bon fonctionnement des réseaux de neurones :
    - la librairie gennan
    - l'implémentation des réseaux de neurones convolutionnels
    - la gestion des matrices par des tenseurs
    - la manipulation des données MNIST
    - la gestion des filtres appliqués au réseau


# tinyML

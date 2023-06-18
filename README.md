
To use this project, you must make the follow commands:

pip install -r requirements.txt
If you use conda virtual env:

conda env create -f environment.yml
conda activate paraphrase-identification


### La démarche méthodologique du projet s’articule autour des points majeurs suivants :


▪ Etude de variables macroeconomiques (Stationnarisation en appliquant un lissage sur les variables macroeconomiques )
▪ Calcule du taux de croissance anuel de chaque série
▪ Construction de ACF et PACF de chaque série
    - Les tracés de l'autocorrélation (ACF) : Ces tracés mesurent la corrélation linéaire entre une variable et ses décalages (lags) passés. 
    Le premier tracé (ligne pleine) représente les valeurs de l'autocorrélation. 
    Les deux autres tracés (lignes en pointillés) représentent les bornes de l'intervalle de confiance pour chaque décalage. Si une barre d'autocorrélation dépasse les bornes de l'intervalle de confiance, cela suggère qu'elle est statistiquement significative et qu'il y a une corrélation entre la variable et ce décalage.

    - Les tracés de l'autocorrélation partielle (PACF) : Ces tracés mesurent la corrélation linéaire entre une variable et ses décalages passés, tout en éliminant l'effet des décalages intermédiaires. 
    Le premier tracé (ligne pleine) représente les valeurs de l'autocorrélation partielle. 
    Les deux autres tracés (lignes en pointillés) représentent les bornes de l'intervalle de confiance pour chaque décalage. Si une barre d'autocorrélation partielle dépasse les bornes de l'intervalle de confiance, cela suggère qu'elle est statistiquement significative et qu'il y a une corrélation entre la variable et ce décalage.

▪ Calcule de la densité spéctrale des variables macroeconomiques 
▪ Construction d’une série de taux de défaut (𝐷𝑅𝑡) 
▪ Construction d’un modèle économétrique afin de prédire le taux de défaut via des variables
macroéconomiques 
▪ Integration de grands points dans une application streamlit 




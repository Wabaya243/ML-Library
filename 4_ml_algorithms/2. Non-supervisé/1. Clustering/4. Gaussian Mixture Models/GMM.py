# Importation des librairies nécessaires
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Charger le dataset
dataset = pd.read_csv('Data/CC GENERAL.csv')

# Suppression de l’identifiant client car il n’apporte rien à la segmentation
dataset = dataset.drop('CUST_ID', axis=1)

# Remplir les valeurs manquantes par la dernière valeur connue (forward fill)
dataset.fillna(method= 'ffill', inplace=True)

# Récupération des colonnes (utile pour vérifier plus tard si besoin)
cols = dataset.columns


# --- Prétraitement des données ---

# Standardisation : transformer les données pour qu’elles aient moyenne=0 et variance=1
scaler = StandardScaler()
scaled_df = scaler.fit_transform(dataset)

# Normalisation : ramener chaque vecteur à une norme unitaire (longueur = 1)
normalised = normalize(scaled_df)

# Conversion en DataFrame pandas
normalised = pd.DataFrame(normalised)


# --- Réduction de dimensionnalité ---

# PCA pour projeter les données en 2 dimensions principales
pca = PCA(n_components=2)
x_principal = pca.fit_transform(normalised)

# Conversion en DataFrame avec noms de colonnes
x_principal = pd.DataFrame(x_principal)
x_principal.columns = {"P1", "P2"}


# --- Création du modèle GMM ---

# Modèle de mélange gaussien avec 3 clusters
gmm = GaussianMixture(n_components=3)
gmm.fit(x_principal)


# --- Visualisation des clusters trouvés par GMM ---

plt.scatter(
    x=x_principal['P1'], 
    y=x_principal['P2'], 
    c=GaussianMixture(n_components=3).fit_predict(x_principal),  # prédiction des clusters
    cmap=plt.cm.winter, 
    alpha=0.6
)
plt.show()


# --- Fonction utilitaire pour sélectionner les meilleures valeurs (plus petites distances) ---
def SelBest(arr:list, X:int)->list:
    '''
    Retourne les X plus petites valeurs dans arr
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


# --- Évaluation par le score de Silhouette ---
n_clusters=np.arange(2, 8)  # tester entre 2 et 7 clusters
sils=[]
sils_err=[]
iterations=20

for n in n_clusters:
    tmp_sil=[]
    for _ in range(iterations):
        gmm=GaussianMixture(n, n_init=2).fit(x_principal) 
        labels=gmm.predict(x_principal)  # attribution des clusters
        sil=metrics.silhouette_score(x_principal , labels, metric='euclidean')
        tmp_sil.append(sil)
    val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))  # moyenne des meilleurs scores
    err=np.std(tmp_sil)  # écart-type
    sils.append(val)
    sils_err.append(err)

# Affichage du score silhouette
plt.errorbar(n_clusters, sils, yerr=sils_err)
plt.title("Silhouette Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")


# --- Fonction pour calculer la distance de Jensen-Shannon entre deux GMMs ---
# (mesure de similarité entre deux modèles de distributions)
def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


# --- Distance entre GMMs entraînés sur train/test ---

n_clusters=np.arange(2, 8)
iterations=20
results=[]
res_sigs=[]

for n in n_clusters:
    dist=[]
    for iteration in range(iterations):
        train, test=train_test_split(x_principal , test_size=0.5)
        gmm_train=GaussianMixture(n, n_init=2).fit(train) 
        gmm_test=GaussianMixture(n, n_init=2).fit(test) 
        dist.append(gmm_js(gmm_train, gmm_test))
    selec=SelBest(np.array(dist), int(iterations/5))
    result=np.mean(selec)
    res_sig=np.std(selec)
    results.append(result)
    res_sigs.append(res_sig)
    
    
# Affichage
plt.errorbar(n_clusters, results, yerr=res_sigs)
plt.title("Distance between Train and Test GMMs", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Distance")
plt.show()


# --- BIC (Bayesian Information Criterion) ---
# Permet de comparer les modèles en pénalisant la complexité (nombre de paramètres)

n_clusters=np.arange(2, 8)
bics=[]
bics_err=[]
iterations=20
for n in n_clusters:
    tmp_bic=[]
    for _ in range(iterations):
        gmm=GaussianMixture(n, n_init=2).fit(x_principal) 
        tmp_bic.append(gmm.bic(x_principal))
    val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
    err=np.std(tmp_bic)
    bics.append(val)
    bics_err.append(err)


# Affichage du BIC
plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
plt.title("BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")
plt.legend()

# Gradient du BIC (permet de voir où il y a une chute nette → bon choix du cluster)
plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
plt.title("Gradient of BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("grad(BIC)")
plt.legend()

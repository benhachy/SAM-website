import numpy as np  # Large, multi-dimensional arrays and matrices manipulations
import pickle

from Modules_spektral_custom.graph import Graph
from Modules_spektral_custom.dataset import Dataset
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error

dir_path = '../ML/Full/'

# Load the dictionary of relations:
dic_edge_num_to_label = {0:"être", 1:"développer/permettre", 2:"avoir", 3:"proposer/offrir", 4:"réaliser/permettre",
                         5:"présenter/illustrer", 6:"analyser/étudier", 7:"nécessiter/devoir", 8:"implémenter/calculer",
                         9:"non être", 10:"non développer/permettre", 11:"non avoir", 12:"non proposer/offrir", 13:"non réaliser/permettre",
                         14:"non présenter/illustrer", 15:"non analyser/étudier", 16:"non nécessiter/devoir", 17:"non implémenter/calculer"}
dic_edge_label_to_num = {v: k for k, v in dic_edge_num_to_label.items()}

# Load entities' names dictionary:
dic_entities_num_to_label = {
     1: 'contrôle', 2: 'processus', 3: 'établissements', 4: 'déplacements', 5: 'cible',
     6: 'réseaux', 7: 'matière', 8: 'machine', 9: 'identification', 10: 'installation',
     11: 'conducteur', 12: 'objectif', 13: 'commande', 14: 'achats', 15: 'interaction',
     16: 'fonction', 17: 'amont', 18: 'variations', 19: 'facturation', 20: 'environnement',
     21: 'véhicules', 22: 'vente', 23: 'enseignant/praticien', 24: 'utilisation', 25: 'magasin',
     26: 'disque/outils', 27: 'surface', 28: 'pièce', 29: 'diversité', 30: 'évaluation',
     31: 'particules', 32: 'logistique', 33: 'intégration', 34: 'intervention', 35: 'consommateurs/usagers',
     36: 'accès/droits', 37: 'paquet', 38: 'aspect', 39: 'événement', 40: 'réaction chimique',
     41: 'liquides', 42: 'information', 43: 'enjeu', 44: 'bâtiments', 45: 'variations',
     46: 'inspection', 47: 'incertitude', 48: 'entreprise', 49: 'démontage/mixage', 50: 'cohérence/fluidité',
     0: ''}
dic_entities_label_to_num = {v: k for k, v in dic_entities_num_to_label.items()}

loss_weight = 10000

# Load the entity classifier:
open_file = open(dir_path + "Modeles_et_fichiers_intermediaires/vocabulary", "rb")
vocab = pickle.load(open_file)
open_file.close()


def tokenize_text(text):
    max_size = 15497
    X = np.zeros((1, max_size))
    for i, word in enumerate(text[:max_size]):
        if word in vocab:
            X[0, i] = vocab[word]
        else:
            X[0, i] = vocab['<UNK>']  # If the word isn't in the vocabulary, take the '<UNK>' token
    return X


def loss(y_true, y_pred):
    return binary_crossentropy(y_true[:, 0], y_pred[:, 0])*loss_weight + mean_absolute_error(y_true[:, 1], y_pred[:, 1])


# Define our graphs with spektral format
class Graphs_Dataset(Dataset):
    def __init__(self, relations, X, **kwargs):
        self.relations = relations
        self.X = X
        super().__init__(**kwargs)

    def read(self):
        # Define our graphs with spektral format:
        n_nodes, n_edges = 51, 18

        e = np.zeros((len(self.relations), n_edges))
        a = np.zeros((n_nodes, n_nodes))
        # 1. Node features: node categories
        x = np.eye(n_nodes)
        for i, (sbj, edges, obj) in enumerate(zip(self.relations["sbj"], self.relations["edges"], self.relations["obj"])):
            sbj = dic_entities_label_to_num[sbj]
            obj = dic_entities_label_to_num[obj]
            for edge in edges:
                # 2. Adjacency matrix:
                a[sbj, obj] += 1
                # 3. Edge attributes: edge categorie
                j = dic_edge_label_to_num[edge]
                e[i, j] += 1
        e =e.astype("float64")
        # 4. Graph label :
        y = []
        # 5. Graph features :
        f = self.X
        return [Graph(x=x, a=a, e=e, y=y, f=f)]

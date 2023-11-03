import os
import click
import pickle
import pandas as pd
import networkx as nx
import multiprocessing as mp
from src.data_matching.EmbDI.edgelist import EdgeList
from src.data_matching.EmbDI.utils import read_edgelist
from src.data_matching.EmbDI.graph import graph_generation
from src.data_matching.EmbDI.sentence_generation_strategies import (
    random_walks_generation,
)
from src.data_matching.EmbDI.embeddings import learn_embeddings
from transformers import BartTokenizer, BartModel, BartConfig
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    RobertaTokenizer,
    RobertaModel,
    GPT2Tokenizer,
    GPT2Model,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
from src.data_matching.data_matching.main_function import (
    parallel_detect_similar_attributes,
)


# model_dict = {
#     'distilbert': DistilBertModel.from_pretrained('distilbert-base-uncased'),
#     'roberta': RobertaModel.from_pretrained('roberta-base'),
#     'gpt2': GPT2Model.from_pretrained('gpt2'),
#     'bert-auto': AutoModel.from_pretrained("bert-base-uncased"),
#     'bert': BertModel.from_pretrained("bert-base-uncased"),
#     'bart': AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli"),

# }
# # Create a dictionary to map tokenizer names to tokenizer classes
# tokenizer_dict = {
#     'distilbert': DistilBertModel.from_pretrained('distilbert-base-uncased'),
#     'roberta': RobertaTokenizer.from_pretrained('roberta-base'),
#     'gpt2': GPT2Tokenizer.from_pretrained('gpt2'),
#     'bert-auto': AutoTokenizer.from_pretrained("bert-base-uncased"),
#     'bert': BertTokenizer.from_pretrained("bert-base-uncased"),
#     'bart': AutoTokenizer.from_pretrained("facebook/bart-large-mnli"),
# }


def edgelist(input_file, out_dir, export, dry_run):
    """Traduit un fichier CSV d'entrée en une liste d'arêtes (edgelist)."""
    # Récupérer le chemin du fichier CSV d'entrée
    dfpath = input_file

    # Déterminer le nom de base pour le fichier de liste d'arêtes (edgelist)
    base_name = os.path.basename(input_file).replace(".csv", ".txt")
    edgefile = os.path.join(out_dir, base_name)

    # Lecture du fichier CSV
    df = pd.read_csv(dfpath, low_memory=False)

    # Préfixes
    pref = ["3#__tn", "3$__tt", "5$__idx", "1$__cid"]

    # Créer la liste d'arêtes (EdgeList)
    el = EdgeList(df, edgefile, pref, None, flatten=True)

    if dry_run:
        if export:
            el.convert_to_dict()
            gdict = el.convert_to_dict()

            # Création d'un graphe NetworkX
            g_nx = nx.from_dict_of_lists(gdict)

            # Création de noms de fichiers pour le graphe NetworkX et le dictionnaire
            n, _ = os.path.splitext(edgefile)
            nx_fname = n + ".nx"
            pkl_fname = n + ".pkl"

            if os.path.exists(nx_fname) and os.path.exists(pkl_fname):
                click.echo(
                    f"{nx_fname} et {pkl_fname} existent déjà. Utilisez l'option --overwrite pour écraser."
                )
            else:
                with open(nx_fname, "wb") as nx_file:
                    pickle.dump(g_nx, nx_file)
                with open(pkl_fname, "wb") as pkl_file:
                    pickle.dump(gdict, pkl_file)

    return edgefile


def Randomwalk(walk_strategy, walk_length, edgelist_file, out_dir):
    """
    Génère des marches aléatoires pour un fichier d'entrée donné selon la stratégie de marche spécifiée.

    Args:
        walk_strategy (str): La stratégie de marche à utiliser.
        walk_length (int): La longueur des phrases de marche.
        edgelist_file (str): Le fichier d'entrée contenant les informations du graphe.

    Returns:
        str: Le nom du fichier généré contenant les marches aléatoires.
    """
    basename = os.path.basename(edgelist_file).replace(".txt", "")
    configuration = {
        "walks_strategy": walk_strategy,
        "flatten": "all",
        "input_file": edgelist_file,
        "n_sentences": "default",
        "sentence_length": walk_length,
        "write_walks": True,
        "intersection": False,
        "backtrack": True,
        "output_file": os.path.join(
            out_dir, basename + f"_{walk_length}_{walk_strategy}"
        ),
        "repl_numbers": False,
        "repl_strings": False,
        "follow_replacement": False,
        "mlflow": False,
    }
    prefixes, edgelist = read_edgelist(configuration["input_file"])
    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration["n_sentences"] == "default":
        # Calcul du nombre de phrases en suivant la règle empirique
        configuration["n_sentences"] = graph.compute_n_sentences(
            int(configuration["sentence_length"])
        )
    walks = random_walks_generation(configuration, graph)
    return configuration["output_file"] + ".walks"


def embdi(
    ndim, window_size, training_algorithm, learning_method, randomwalk_file, out_dir
):
    """
    Utilise l'algorithme EMBDI pour apprendre les embeddings des données à partir des marches aléatoires générées.

    Args:
        ndim (int): La dimension des embeddings à apprendre.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.
        randomwalk_file (str): Le fichier contenant les marches aléatoires.

    Returns:
        None
    """
    name_dataset_file = os.path.basename(randomwalk_file).replace(".walks", "")
    file_name_dataset = os.path.join(out_dir, name_dataset_file + ".emb")
    with open(file_name_dataset, "w") as file:
        file.write("")
    output_embeddings_file = os.path.join(file_name_dataset)
    walks = randomwalk_file
    write_walks = True
    learn_embeddings(
        output_embeddings_file,
        walks,
        write_walks,
        ndim,
        window_size,
        training_algorithm=training_algorithm,
        learning_method=learning_method,
        workers=mp.cpu_count(),
        sampling_factor=0.001,
    )
    return file_name_dataset


def detect_similarity(embdi_s1_file, attributes_s2, model, tokenizer):
    attributes_s2 = ["gender", "dateofbirth", "first", "ticket"]
    configuration = BartConfig(d_model=32)
    model = BartModel(configuration)
    configuration = model.config
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    precision, recall, f1_score = parallel_detect_similar_attributes(
        embdi_s1_file, attributes_s2, model, tokenizer
    )
    click.echo(f"Precision: {precision}")
    click.echo(f"Recall: {recall}")
    click.echo(f"F1 Score: {f1_score}")

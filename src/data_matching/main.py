import click
import os
import logging
import networkx as nx
import pandas as pd
import pickle
from src.data_matching.__init__ import __version__
from src.data_matching.EmbDI.edgelist import EdgeList
from src.data_matching.EmbDI.utils import read_edgelist
from src.data_matching.EmbDI.graph import graph_generation
from src.data_matching.EmbDI.sentence_generation_strategies import random_walks_generation

# Configuration du logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    level=logging.INFO,
)

@click.group()
def cli():
    """Fonction CLI racine."""
    pass

@cli.command()
def version():
    """Affiche les informations sur la version."""
    click.echo(__version__)

@cli.command()
@click.option(
    "-i", "--input",
    "input_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Chemin vers le fichier CSV d'entrée à traduire.",
)
@click.option(
    "-o", "--output",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Répertoire de sortie pour le fichier de liste d'arêtes (edgelist).",
)
@click.option(
    "--export", "-e",
    "export",
    type=(str, float),
    multiple=True,
    help="Drapeau pour exporter la liste d'arêtes au format NetworkX.",
)
@click.option(
    "--edgelist-file", "-egf",
    "edgelist_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    multiple=True,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "--walk-strategy", "-ws",
    "walk_strategy",
    type=str,
    multiple=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--walk-length", "-wl",
    "walk_length",
    type=int,
    default=0,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--dry-run",
    "dry_run",
    type=bool,
    is_flag=True,
    default=False,
    help="Simulation, n'écrira rien.",
)
def get_edgelist(input_file, out_dir, export, edgelist_file, walk_strategy, walk_length, dry_run):
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
                click.echo(f"{nx_fname} et {pkl_fname} existent déjà. Utilisez l'option --overwrite pour écraser.")
            else:
                with open(nx_fname, "wb") as nx_file:
                    pickle.dump(g_nx, nx_file)
                with open(pkl_fname, "wb") as pkl_file:
                    pickle.dump(gdict, pkl_file)

    return edgefile

@cli.command()
@click.option(
    "--walk-strategy", "-ws",
    "walk_strategy",
    type=str,
    required=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--walk-length", "-wl",
    "walk_length",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--edgelist-file", "-ef",
    "edgelist_file",
    type=str,
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "-orw", "--output_rw",
    "out_dir_rw",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Répertoire de sortie pour le fichier des random walks.",
)
def Randomwalk(walk_strategy, walk_length, edgelist_file,out_dir_rw):
    """
    Génère des marches aléatoires pour un fichier d'entrée donné selon la stratégie de marche spécifiée.

    Args:
        walk_strategy (str): La stratégie de marche à utiliser.
        walk_length (int): La longueur des phrases de marche.
        edgelist_file (str): Le fichier d'entrée contenant les informations du graphe.

    Returns:
        str: Le nom du fichier généré contenant les marches aléatoires.
    """
    basename=os.path.basename(edgelist_file).replace(".txt","")
    configuration = {
        'walks_strategy': walk_strategy,
        'flatten': 'all',
        'input_file': edgelist_file,
        'n_sentences': 'default',
        'sentence_length': walk_length,
        'write_walks': True,
        'intersection': False,
        'backtrack': True,
        'output_file': os.path.join(out_dir_rw,basename + f'_{walk_length}_{walk_strategy}'),
        'repl_numbers': False,
        'repl_strings': False,
        'follow_replacement': False,
        'mlflow': False
    }
    prefixes, edgelist = read_edgelist(configuration['input_file'])
    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration['n_sentences'] == 'default':
        # Calcul du nombre de phrases en suivant la règle empirique
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    walks = random_walks_generation(configuration, graph)
    return configuration["output_file"]+".walks"

if __name__ == "__main__":
    cli()

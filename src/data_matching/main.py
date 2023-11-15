import click
import logging
import os
import pandas as pd
from src.data_matching.__init__ import __version__
from src.data_matching.main_main import edgelist, Randomwalk, embdi, detect_similarity
from transformers import BartTokenizer, BartModel, BartConfig
from src.data_matching.data_matching.main_function import (
    parallel_detect_similar_attributes,
)


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
    "-i",
    "--input",
    "input_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Chemin vers le fichier CSV d'entrée à traduire.",
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Répertoire de sortie pour le fichier de liste d'arêtes (edgelist).",
)
@click.option(
    "--export",
    "-e",
    "export",
    type=bool,
    is_flag=True,
    default=False,
    help="Drapeau pour exporter la liste d'arêtes au format NetworkX.",
)
@click.option(
    "--dry-run",
    "dry_run",
    type=bool,
    is_flag=True,
    default=False,
    help="Simulation, n'écrira rien.",
)
def edgelist_(input_file, out_dir, export, dry_run):
    return edgelist(input_file, out_dir, export, dry_run)


@cli.command()
@click.option(
    "--walk-strategy",
    "-ws",
    "walk_strategy",
    type=click.Choice(["basic", "node2vec", "deepwalk", "metapath2vec"]),
    required=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--walk-length",
    "-wl",
    "walk_length",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--edgelist-file",
    "-i",
    "edgelist_file",
    type=str,
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Répertoire de sortie pour le fichier des random walks.",
)
def Randomwalk_(walk_strategy, walk_length, edgelist_file, out_dir):
    return Randomwalk(walk_strategy, walk_length, edgelist_file, out_dir)


@cli.command()
@click.option(
    "--ndim",
    "-nd",
    "ndim",
    type=int,
    required=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--window-size",
    "-wis",
    "window_size",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--training-algorithm",
    "-ta",
    "training_algorithm",
    type=click.Choice(["word2vec", "fasttext","doc2vec"]),
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "--learning-method",
    "-lm",
    "learning_method",
    type=click.Choice(["skipgram", "CBOW"]),
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "--input",
    "-i",
    "randomwalk_file",
    type=str,
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Répertoire de sortie pour le fichier des embeddings.",
)
def embdi_(
    ndim, window_size, training_algorithm, learning_method, randomwalk_file, out_dir
):
    return embdi(
        ndim, window_size, training_algorithm, learning_method, randomwalk_file, out_dir
    )


@cli.command()
@click.option(
    "--input",
    "-i",
    "embdi_s1_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Chemin vers le fichier d'embdi S1.",
)
@click.option(
    "--attributes-s2",
    "-as2",
    "attributes_s2",
    type=str,
    required=False,
    multiple=True,
    help="Liste des attributs S2.",
)
@click.option(
    "--model",
    "-m",
    "model",
    type=click.Choice(["distilbert", "roberta", "gpt2", "bert", "bert-auto", "bart"]),
    required=True,
    help="Choix du modèle.",
)
@click.option(
    "--tokenizer",
    "-t",
    "tokenizer",
    type=click.Choice(["distilbert", "roberta", "gpt2", "bert", "bert-auto", "bart"]),
    required=True,
    help="Choix du tokenizer.",
)
def detect_similarity_(embdi_s1_file, attributes_s2, model, tokenizer):
    return detect_similarity(embdi_s1_file, attributes_s2, model, tokenizer)


@cli.command()
@click.option(
    "--input1",
    "-i1",
    "input_file1",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Chemin vers le fichier d'embdi S1.",
)
@click.option(
    "--input2",
    "-i2",
    "input_file2",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Chemin vers le fichier d'embdi S1.",
)
@click.option(
    "--walk-strategy",
    "-ws",
    "walk_strategy",
    type=click.Choice(["basic", "node2vec", "deepwalk", "metapath2vec"]),
    required=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--n-sentences",
    "-ns",
    "n_sentences",
    type=int,
    required=True,
    help="nombre de phrase de la marche aléatoire.",
)
@click.option(
    "--walk-length",
    "-wl",
    "walk_length",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--ndim",
    "-nd",
    "ndim",
    type=int,
    required=True,
    help="Stratégie de marche pour les marches aléatoires.",
)
@click.option(
    "--window-size",
    "-w",
    "window_size",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--training-algorithm",
    "-ta",
    "training_algorithm",
    type=click.Choice(["word2vec", "fasttext","doc2vec"]),
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)
@click.option(
    "--learning-method",
    "-lm",
    "learning_method",
    type=click.Choice(["skipgram", "CBOW"]),
    required=False,
    help="Chemin vers le fichier de liste d'arêtes.",
)

def detect_similarity_all(input_file1, input_file2,walk_strategy,walk_length,ndim,window_size,training_algorithm,learning_method,n_sentences):
    name=os.path.basename(input_file1).replace("_source.csv", "")
    if not os.path.exists(f"/tests/output_data/data_frames/{name}/{name}_{walk_strategy}_{walk_length}_{n_sentences}_{ndim}_{window_size}_{training_algorithm}_{learning_method}_{n_sentences}.csv"):
        attributes_s2 = list(pd.read_csv(input_file2).columns)
        edgelist_file = edgelist(input_file1, "./tests/output_data/edglistes", False, False)
        randomwalk_file = Randomwalk(
            walk_strategy=walk_strategy,
            walk_length=walk_length,
            n_sentences=n_sentences,
            edgelist_file=edgelist_file,
            out_dir="./tests/output_data/walks",
        )
        embdi_s1_file = embdi(
            ndim=ndim,
            window_size=window_size,
            training_algorithm=training_algorithm,
            learning_method=learning_method,
            randomwalk_file=randomwalk_file,
            out_dir="./tests/output_data/embbedings",
        )
        configuration = BartConfig(d_model=ndim)
        model = BartModel(configuration)
        configuration = model.config
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        precision, recall, f1_score,M_s1Emb,M_s1Mod,M_S2Emb,M_S2Mod = parallel_detect_similar_attributes(
            embdi_s1_file, attributes_s2, model, tokenizer
        )
        new_row = {'dataset': name,'walk_strategy':walk_strategy, 'walk_length':walk_length,'dimension':ndim,'window':window_size,'learning_methd':learning_method,'training_algorithms':training_algorithm,'n_sentences':n_sentences,'F1_score':f1_score,'recall':recall,'precision':precision}
        data_s1={"M_s1Emb":M_s1Emb,"M_s1Mod":M_s1Mod}
        data_s2={"M_s2Emb":M_S2Emb,"M_s2Mod":M_S2Mod}
        result = f"./tests/output_data/data_frames/{name}/{name}_{walk_strategy}_{walk_length}_{n_sentences}_{ndim}_{window_size}_{training_algorithm}_{learning_method}_{n_sentences}"
        pd.DataFrame(new_row, index=[0], columns=['dataset', 'walk_strategy', 'walk_length', 'dimension', 'window', 'learning_methd',"training_algorithms","n_sentences","F1_score","recall","precision"]).to_csv(f"{result}.csv")
        # pd.DataFrame(data_s1, index=[0]).to_json(f"{result}_s1.json")
        # pd.DataFrame(data_s2, index=[0]).to_json(f"{result}_s2.json")
        click.echo(f"Precision: {precision}")
        click.echo(f"Recall: {recall}")
        click.echo(f"F1 Score: {f1_score}")


if __name__ == "__main__":
    cli()

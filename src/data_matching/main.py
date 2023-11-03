import click
import logging
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
    "-ws",
    "window_size",
    type=int,
    required=True,
    help="Longueur de la marche aléatoire.",
)
@click.option(
    "--training-algorithm",
    "-ta",
    "training_algorithm",
    type=click.Choice(["word2vec", "fasttext"]),
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
def detect_similarity_all(input_file1, input_file2):
    attributes_s2 = list(pd.read_csv(input_file2).columns)
    edgelist_file = edgelist(input_file1, "./tests/output_data/edglistes", False, False)
    randomwalk_file = Randomwalk(
        walk_strategy="deepwalk",
        walk_length=5000,
        edgelist_file=edgelist_file,
        out_dir="./tests/output_data/walks",
    )
    embdi_s1_file = embdi(
        ndim=64,
        window_size=3,
        training_algorithm="word2vec",
        learning_method="skipgram",
        randomwalk_file=randomwalk_file,
        out_dir="./tests/output_data/embbedings",
    )
    configuration = BartConfig(d_model=64)
    model = BartModel(configuration)
    configuration = model.config
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    precision, recall, f1_score = parallel_detect_similar_attributes(
        embdi_s1_file, attributes_s2, model, tokenizer
    )
    click.echo(f"Precision: {precision}")
    click.echo(f"Recall: {recall}")
    click.echo(f"F1 Score: {f1_score}")


if __name__ == "__main__":
    cli()

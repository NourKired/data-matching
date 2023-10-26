import click
import src.data_matching.__init__ as init
from src.data_matching.EmbDI.edgelist import EdgeList
import os.path as osp
import pickle
import logging
import networkx as nx
import pandas as pd
import os 

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    level=logging.INFO,
)

@click.group()
def cli():
    """Root CLI function."""
    pass

@cli.command()
def version():
    """Display the version information."""
    click.echo(init.__version__)

@cli.command()
@click.option(
    "-i", "--input",
    "input_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, readable=True),
    required=True,
    help="Path to input CSV file to translate.",
)
@click.option(
    "-o", "--output",
    "out_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, readable=True),
    default=".",
    show_default=True,
    help="Output directory for edgelist file.",
)
@click.option(
    "--export", "-e",
    "export",
    type=(str, float),
    multiple=True,
    help="Flag for exporting the edgelist in NetworkX format.",
)
@click.option(
    "-f", "--force",
    "overwrite",
    type=bool,
    is_flag=True,
    default=False,
    help="Overwrite existing files.",
)
@click.option(
    "--dry-run",
    "dry_run",
    type=bool,
    is_flag=True,
    default=False,
    help="Pass through, will not write anything.",
)
def get_edgelist(input_file, out_dir, export: bool =True, overwrite: bool = True, dry_run: bool = False):
    """Translate an input CSV file into an edgelist."""
    print("heeeyyy")
    dfpath = input_file
    base_name = os.path.basename(input_file).replace(".csv", ".txt")
    print("base_name",base_name)
    edgefile = os.path.join(out_dir, base_name)
    info_file = None
    df = pd.read_csv(dfpath, low_memory=False)
    pref = ["3#__tn", "3$__tt", "5$__idx", "1$__cid"]
    el = EdgeList(df, edgefile, pref, info_file, flatten=True)
    if dry_run:
        if export:
            el.convert_to_dict()
            gdict = el.convert_to_dict()
            print("el", el.convert_to_dict())
            print("gdict", gdict)
            g_nx = nx.from_dict_of_lists(gdict)
            n, _ = osp.splitext(edgefile)
            nx_fname = n + ".nx"
            pkl_fname = n + ".pkl"
            if overwrite:
                with open(nx_fname, "wb") as nx_file:
                    pickle.dump(g_nx, nx_file)
                with open(pkl_fname, "wb") as pkl_file:
                    pickle.dump(gdict, pkl_file)

if __name__ == "__main__":
    cli()

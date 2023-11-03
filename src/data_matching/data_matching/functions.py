import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BartTokenizer, BartModel, BartConfig
from transformers import BartConfig, BartModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_emb_model(model, tokenizer, text):
    """
    Récupère les embeddings d'un texte à partir d'un modèle pré-entraîné.

    Args:
        model (torch model): Le modèle pré-entraîné.
        tokenizer (tokenizer): Le tokenizer associé au modèle.
        text (str): Le texte à vectoriser.

    Returns:
        list: Les embeddings du texte.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])[0]
    return embeddings.detach().numpy()


def get_embeddings_Embdi(Embdi_S1_file, passage=True):
    """
    Récupère les embeddings des données à partir des résultats de l'algorithme EMBDI.

    Args:
        filepath (str): Le chemin du fichier contenant les embeddings.

    Returns:
        list: Les labels des attributs S1.
        np.array: Les embeddings des attributs S1.
    """
    S1_embeddings = []
    S1_attributes = []
    with open(Embdi_S1_file, "r") as file:
        for i, line in enumerate(file):
            if i != 0:
                values = line.strip().split()
                node = values[0]
                if passage:
                    if node.startswith("tt"):
                        S1_attributes.append(node[4:])
                        S1_embeddings.append(values[1:])
                if node.startswith("cid"):
                    S1_attributes.append(node[5:])
                    S1_embeddings.append(values[1:])
    S1_embeddings = np.asarray(S1_embeddings, dtype=float)
    S1_embeddings.astype(float)
    return S1_attributes, S1_embeddings


def get_embeddings_S1(Embdi_S1_file, model, tokenizer, passage=True):
    """
        Récupère les embeddings des attributs de l'ontologie et du dataset à partir des résultats de l'algorithme EMBDI.

        Args:
    ts dans EMBDI.
            ndim (int): La dimension des embeddings appris.
            Embdi_S1_file (str): Le chemin du fichier contenant les embeddings S1 attributes.

        Returns:
            tuple: Un tuple contenant les embeddings, les vrais labels, les labels prédits et la matrice d'embedding.
    """
    attributes_S1, M_s1Emb = get_embeddings_Embdi(Embdi_S1_file, passage=False)
    data_passage, M_s1Emb_passage = get_embeddings_Embdi(Embdi_S1_file, passage=False)
    ndim = len(M_s1Emb[0])
    M_s1Mod = []
    M_s1Mod_passage = []
    for att_s1 in attributes_S1:
        emb_att_s1 = get_emb_model(model, tokenizer, att_s1)
        M_s1Mod.append(list(emb_att_s1))

    for data in data_passage:
        emb_data_s1 = get_emb_model(model, tokenizer, data)
        M_s1Mod_passage.append(list(emb_data_s1))

    M_s1Emb = np.array(M_s1Emb).astype(float)
    M_s1Mod_passage = np.array(M_s1Mod_passage).astype(float)
    return M_s1Emb, M_s1Mod, attributes_S1, M_s1Emb_passage, M_s1Mod_passage


def get_M_S2Mod_embeddings(attributes_S2, model, tokenizer):
    M_s2Mod = []
    for att_s2 in attributes_S2:
        v_att_s2 = np.array(get_emb_model(model, tokenizer, att_s2))
        M_s2Mod.append(v_att_s2)
    return np.array(M_s2Mod)


def get_transformed_embeddings(embeddings, pca):
    return pca.transform(embeddings)


def get_dot_product(matrix, other_matrix, transpose=False):
    if transpose:
        return np.dot(matrix.T, other_matrix)
    return np.dot(matrix, other_matrix)


def get_nmax(liste, n):
    nliste = sorted(liste, reverse=True)
    return nliste[:n]


def find_similar_attributes(M_s1Emb, M_s2emb, attributes_S2, selected_att_S1):
    all_detected = []
    y_pred = []
    for v_att, att_s1 in zip(M_s1Emb, selected_att_S1):
        print(att_s1)
        cos_sim_values = cosine_similarity(
            np.array(v_att).reshape(1, -1), np.array(M_s2emb)
        )[0]
        indices = [
            i
            for i, sim in enumerate(cos_sim_values)
            if np.abs(sim) >= 0.5 and sim in get_nmax(cos_sim_values, 2)
        ]
        detected_S2 = [
            attributes_S2[i] for i in indices if "Nan" not in attributes_S2[i]
        ]
        print(detected_S2)
        all_detected.append(detected_S2)
        y_pred.append(1 if att_s1 in detected_S2 else 0)

    return detected_S2, y_pred


def create_y_true(selected_S2, attributes_S1):
    y_true = []
    for att_s2 in selected_S2:
        y_true.append(1 if att_s2 in attributes_S1 else 0)
    return y_true


def Matrice_de_passage(A1, A2):
    """
    Calcule la matrice de passage entre deux ensembles de données dans un espace vectoriel.

    Args:
        A1 (np.array): Le premier ensemble de données.
        A2 (np.array): Le deuxième ensemble de données.

    Returns:
        np.array: La matrice de passage.
        np.array: Les vecteurs propres de A1.
        np.array: Les vecteurs propres de A2.
        np.array: Les valeurs singulières de A1.
        np.array: Les valeurs singulières de A2.
    """
    U1, s1, V1 = np.linalg.svd(A1)
    U2, s2, V2 = np.linalg.svd(A2)
    matrice_passageu1_u2 = np.dot(A1.T, A2)
    return matrice_passageu1_u2, U1, U2, V1, V2, s1, s2

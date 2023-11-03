import numpy as np
from src.data_matching.data_matching.functions import (
    get_embeddings_S1,
    get_transformed_embeddings,
    find_similar_attributes,
    create_y_true,
    get_M_S2Mod_embeddings,
    get_dot_product,
    Matrice_de_passage,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import Parallel, delayed


def process_iteration(
    it,
    attributes_s2_new,
    attributes_S1,
    model,
    tokenizer,
    U1,
    P,
    V2,
    diff,
    M_s1Emb,
    M_S2Emb,
    M_S2Mod,
):
    attributes_S2_it = attributes_s2_new[
        it * len(attributes_S1) : (it + 1) * len(attributes_S1)
    ]
    M_s2Mod = get_M_S2Mod_embeddings(attributes_S2_it, model, tokenizer)
    # M_s2Mod_svm = get_dot_product(M_s2Mod, U1, transpose=True)
    # print(M_s2Mod_svm.shape)
    M_s2Emb_svm = get_dot_product(M_s2Mod, P)
    # M_s2Emb = get_dot_product(M_s2Emb_svm, V2, transpose=True)
    # print(M_s2Mod.shape)
    detected_S2, y_pred = find_similar_attributes(
        M_s1Emb, M_s2Emb_svm, attributes_S2_it, attributes_S1
    )
    y_true = create_y_true(attributes_S2_it, attributes_S1)
    M_S2Emb.append(M_s2Emb_svm)
    M_S2Mod.append(M_s2Mod)
    # if it == -1:
    #     M_S2Emb = M_S2Emb[:-diff]
    #     M_S2Mod = M_S2Mod[:-diff]
    precision, recall, f1_Score = (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )
    results = np.round(precision, 2), np.round(recall, 2), np.round(f1_Score, 2)
    return results


def parallel_detect_similar_attributes(Embdi_S1_file, attributes_s2, model, tokenizer):
    (
        M_s1Emb,
        M_s1Mod,
        attributes_S1,
        M_s1Emb_passage,
        M_s1Mod_passage,
    ) = get_embeddings_S1(Embdi_S1_file, model, tokenizer)
    # attributes_s2=attributes_S1
    P, U1, U2, V1, V2, s1, s2 = Matrice_de_passage(
        A1=M_s1Mod_passage, A2=M_s1Emb_passage
    )
    # indexes_to_remove = np.random.randint(0, len(attributes_S1), size=0)
    # not_selected_S2 = [attributes_S1[i] for i in indexes_to_remove]
    # selected_S2 = [attributes_S1[i] for i in range(len(attributes_S1)) if i not in indexes_to_remove]
    # attributes_S2 = [att_s2 for att_s2 in attributes_s2 if att_s2 not in not_selected_S2]
    diff = len(attributes_S1) - int(len(attributes_s2) % len(attributes_S1))
    M_S2Emb = []
    M_S2Mod = []
    if diff < len(attributes_S1):
        attributes_s2_new = attributes_s2 + ["Nan"] * diff
        precisions, recalls, f1_Scores = [], [], []
        results = Parallel(n_jobs=-1)(
            delayed(process_iteration)(
                it,
                attributes_s2_new,
                attributes_S1,
                model,
                tokenizer,
                U1,
                P,
                V2,
                diff,
                M_s1Emb,
                M_S2Emb,
                M_S2Mod,
            )
            for it in range(len(attributes_s2) // len(attributes_S1) + 1)
        )
        precision, recall, f1_Score = zip(*results)
        precisions.append(precision)
        recalls.append(recall)
        f1_Scores.append(f1_Score)
        recisions, recalls, f1_Scores = (
            np.mean(precisions),
            np.mean(recalls),
            np.mean(f1_Scores),
        )
    else:
        results = process_iteration(
            0,
            attributes_s2,
            attributes_S1,
            model,
            tokenizer,
            U1,
            P,
            V2,
            diff,
            M_s1Emb,
            M_S2Emb,
            M_S2Mod,
        )
        precisions, recalls, f1_Scores = results

    print(precisions, recalls, f1_Scores)
    return np.mean(precisions), np.mean(recalls), np.mean(f1_Scores)

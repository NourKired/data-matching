import numpy as np
from scipy.optimize import curve_fit
import ast
import pandas as pd


def power_law(x: float, alpha: float, c: float) -> float:
    """
    Calculate the result of a power-law expression.

    Parameters:
    x (float): The base value.
    alpha (float): The exponent.
    c (float): The constant factor.

    Returns:
    float: The result of the power-law expression.
    """
    return c * x ** (-alpha)


def alpha(M: np.ndarray) -> float:
    """
    Calculate the alpha value of a matrix using a power-law fit to its singular values.

    Parameters:
    M (np.ndarray): The input matrix.

    Returns:
    float: The alpha value.
    """
    singular_values = np.linalg.svd(np.array(M), compute_uv=False)
    ranks = np.arange(1, len(singular_values) + 1)
    params, _ = curve_fit(power_law, ranks, singular_values)
    return float(params[0])


def rank_me(embedding_matrix: np.ndarray) -> float:
    """
    Calculate the rank-based entropy of an embedding matrix.

    Parameters:
    embedding_matrix (np.ndarray): The embedding matrix.

    Returns:
    float: The entropy value.
    """
    singular_values = np.linalg.svd(embedding_matrix, compute_uv=False)
    normalized_singular_values = singular_values / np.sum(singular_values)
    entropy = -np.sum(normalized_singular_values * np.log2(normalized_singular_values))
    return entropy


def nesum(embedding_matrix: np.ndarray) -> float:
    """
    Calculate the normalized eigenvalue sum of a covariance matrix derived from the embedding matrix.

    Parameters:
    embedding_matrix (np.ndarray): The embedding matrix.

    Returns:
    float: The normalized eigenvalue sum.
    """
    covariance_matrix = np.cov(embedding_matrix, rowvar=False)
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    nesum_score = np.sum(eigenvalues) / eigenvalues[0]
    return nesum_score


def condition_number(embedding_matrix: np.ndarray) -> float:
    """
    Calculate the condition number of an embedding matrix.

    Parameters:
    embedding_matrix (np.ndarray): The embedding matrix.

    Returns:
    float: The condition number.
    """
    return np.linalg.cond(embedding_matrix)


def stable_rank(embedding_matrix: np.ndarray) -> float:
    """
    Calculate the stable rank of an embedding matrix.

    Parameters:
    embedding_matrix (np.ndarray): The embedding matrix.

    Returns:
    float: The stable rank.
    """
    return np.linalg.norm(embedding_matrix, "fro") / np.linalg.norm(embedding_matrix, 2)


def selfcluster(W: np.array) -> float:
    """
    Calculate the self-cluster metric for a given matrix.

    Args:
        W (np.array): The input matrix.

    Returns:
        float: The self-cluster metric.
    """
    n, d = W.shape
    WWt = np.dot(W, W.T)
    norm_WWt_Frobenius = np.linalg.norm(WWt, "fro")
    return (d * norm_WWt_Frobenius - n * (d + n - 1)) / ((d - 1) * (n - 1) * n)


def calculate_metrics(
    data: np.array,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate various metrics for a given data matrix.

    Args:
        data (np.array): The input data matrix.

    Returns:
        Tuple[float, float, float, float, float, float]: Tuple of alpha, nesum, rank_me, condition_number, stable_rank, and selfcluster metrics.
    """
    try:
        data = ast.literal_eval(data)
    except:
        pass
    data = np.array(data)
    alpha_val = alpha(data)
    nesum_val = nesum(data)
    rank_me_val = rank_me(data)
    condition_number_val = condition_number(data)
    stable_rank_val = stable_rank(data)
    selfcluster_val = selfcluster(data)
    return (
        alpha_val,
        nesum_val,
        rank_me_val,
        condition_number_val,
        stable_rank_val,
        selfcluster_val,
    )


def get_metriques(
    M_D_Emb: np.array, M_D_Mod: np.array, M_E_Emb: np.array, M_E_Mod: np.array
) -> pd.DataFrame:
    """
    Calculate multiple metrics for a set of input matrices and create a DataFrame.

    Args:
        M_D_Emb (np.array): Input matrices for Data in Embdi.
        M_D_Mod (np.array): Input matrices for Data in pre-trained model.
        M_E_Emb (np.array): Input matrices for attribute of schema in Embdi.
        M_E_Mod (np.array): Input matrices for attribute of schema in pre-trained model.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics for embedding stability.
    """
    df_metriques = pd.DataFrame(
        columns=[
            "alpha_M_D_Emb",
            "alpha_M_E_Emb",
            "alpha_M_D_Mod",
            "alpha_M_E_Mod",
            "nesum_M_D_Emb",
            "nesum_M_E_Emb",
            "nesum_M_D_Mod",
            "nesum_M_E_Mod",
            "rank_me_M_D_Emb",
            "rank_me_M_E_Emb",
            "rank_me_M_D_Mod",
            "rank_me_M_E_Mod",
            "condition_number_M_D_Emb",
            "condition_number_M_E_Emb",
            "condition_number_M_D_Mod",
            "condition_number_M_E_Mod",
            "stable_rank_M_D_Emb",
            "stable_rank_M_E_Emb",
            "stable_rank_M_D_Mod",
            "stable_rank_M_E_Mod",
            "selfcluster_M_D_Emb",
            "selfcluster_M_E_Emb",
            "selfcluster_M_D_Mod",
            "selfcluster_M_E_Mod",
        ]
    )

    for i, (m_d_emb, m_d_mod, m_e_emb, m_e_mod) in enumerate(
        zip(M_D_Emb, M_D_Mod, M_E_Emb, M_E_Mod)
    ):
        (
            alpha_m_d_emb,
            nesum_m_d_emb,
            rank_me_m_d_emb,
            condition_number_m_d_emb,
            stable_rank_m_d_emb,
            selfcluster_m_d_emb,
        ) = calculate_metrics(m_d_emb)
        (
            alpha_m_d_mod,
            nesum_m_d_mod,
            rank_me_m_d_mod,
            condition_number_m_d_mod,
            stable_rank_m_d_mod,
            selfcluster_m_d_mod,
        ) = calculate_metrics(m_d_mod)
        (
            alpha_m_e_emb,
            nesum_m_e_emb,
            rank_me_m_e_emb,
            condition_number_m_e_emb,
            stable_rank_m_e_emb,
            selfcluster_m_e_emb,
        ) = calculate_metrics(m_e_emb)
        (
            alpha_m_e_mod,
            nesum_m_e_mod,
            rank_me_m_e_mod,
            condition_number_m_e_mod,
            stable_rank_m_e_mod,
            selfcluster_m_e_mod,
        ) = calculate_metrics(m_e_mod)

        metrics = {
            "alpha_M_D_Emb": alpha_m_d_emb,
            "alpha_M_E_Emb": alpha_m_e_emb,
            "alpha_M_D_Mod": alpha_m_d_mod,
            "alpha_M_E_Mod": alpha_m_e_mod,
            "nesum_M_D_Emb": nesum_m_d_emb,
            "nesum_M_E_Emb": nesum_m_e_emb,
            "nesum_M_D_Mod": nesum_m_d_mod,
            "nesum_M_E_Mod": nesum_m_e_mod,
            "rank_me_M_D_Emb": rank_me_m_d_emb,
            "rank_me_M_E_Emb": rank_me_m_e_emb,
            "rank_me_M_D_Mod": rank_me_m_d_mod,
            "rank_me_M_E_Mod": rank_me_m_e_mod,
            "condition_number_M_D_Emb": condition_number_m_d_emb,
            "condition_number_M_E_Emb": condition_number_m_e_emb,
            "condition_number_M_D_Mod": condition_number_m_d_mod,
            "condition_number_M_E_Mod": condition_number_m_e_mod,
            "stable_rank_M_D_Emb": stable_rank_m_d_emb,
            "stable_rank_M_E_Emb": stable_rank_m_e_emb,
            "stable_rank_M_D_Mod": stable_rank_m_d_mod,
            "stable_rank_M_E_Mod": stable_rank_m_e_mod,
            "selfcluster_M_D_Emb": selfcluster_m_d_emb,
            "selfcluster_M_E_Emb": selfcluster_m_e_emb,
            "selfcluster_M_D_Mod": selfcluster_m_d_mod,
            "selfcluster_M_E_Mod": selfcluster_m_e_mod,
        }

        metrics["alpha"] = (
            sum(
                metrics[k]
                for k in [
                    "alpha_M_D_Emb",
                    "alpha_M_E_Emb",
                    "alpha_M_D_Mod",
                    "alpha_M_E_Mod",
                ]
            )
            / 4
        )
        metrics["nesum"] = (
            sum(
                metrics[k]
                for k in [
                    "nesum_M_D_Emb",
                    "nesum_M_E_Emb",
                    "nesum_M_D_Mod",
                    "nesum_M_E_Mod",
                ]
            )
            / 4
        )
        metrics["rank_me"] = (
            sum(
                metrics[k]
                for k in [
                    "rank_me_M_D_Emb",
                    "rank_me_M_E_Emb",
                    "rank_me_M_D_Mod",
                    "rank_me_M_E_Mod",
                ]
            )
            / 4
        )
        metrics["condition_number"] = (
            sum(
                metrics[k]
                for k in [
                    "condition_number_M_D_Emb",
                    "condition_number_M_E_Emb",
                    "condition_number_M_D_Mod",
                    "condition_number_M_E_Mod",
                ]
            )
            / 4
        )
        metrics["stable_rank"] = (
            sum(
                metrics[k]
                for k in [
                    "stable_rank_M_D_Emb",
                    "stable_rank_M_E_Emb",
                    "stable_rank_M_D_Mod",
                    "stable_rank_M_E_Mod",
                ]
            )
            / 4
        )
        metrics["selfcluster"] = (
            sum(
                metrics[k]
                for k in [
                    "selfcluster_M_D_Emb",
                    "selfcluster_M_E_Emb",
                    "selfcluster_M_D_Mod",
                    "selfcluster_M_E_Mod",
                ]
            )
            / 4
        )

        df_metriques = df_metriques.append(metrics, ignore_index=True)

    return df_metriques

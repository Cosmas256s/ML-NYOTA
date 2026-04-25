from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PARAM_SCHEMA = ["skill", "p_l0", "p_t", "p_g", "p_s", "p_f"]


def _normalize_sequences(responses: Sequence[Sequence[int]] | Sequence[int]) -> list[np.ndarray]:
    """Convert a single response array or a nested collection into clean sequences."""
    if isinstance(responses, (list, tuple)) and responses and isinstance(
        responses[0], (list, tuple, np.ndarray, pd.Series)
    ):
        return [np.asarray(seq, dtype=int) for seq in responses if len(seq) > 0]

    arr = np.asarray(responses, dtype=int)
    return [arr[arr >= 0]] if arr.size else []


def _forward_backward(seq: np.ndarray, params: Sequence[float]) -> dict[str, float | np.ndarray]:
    """Run forward-backward for a single binary BKT sequence."""
    p_l0, p_t, p_g, p_s, p_f = params
    seq = np.asarray(seq, dtype=int)
    if seq.size == 0:
        return {}

    transition = np.array([[1 - p_t, p_t], [p_f, 1 - p_f]], dtype=float)
    emission = np.array([[1 - p_g, p_g], [p_s, 1 - p_s]], dtype=float)
    prior = np.array([1 - p_l0, p_l0], dtype=float)

    t_len = len(seq)
    alpha = np.zeros((t_len, 2), dtype=float)
    scales = np.zeros(t_len, dtype=float)

    alpha[0] = prior * emission[:, seq[0]]
    scales[0] = alpha[0].sum()
    alpha[0] /= max(scales[0], 1e-12)

    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ transition) * emission[:, seq[t]]
        scales[t] = alpha[t].sum()
        alpha[t] /= max(scales[t], 1e-12)

    beta = np.zeros((t_len, 2), dtype=float)
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = transition @ (emission[:, seq[t + 1]] * beta[t + 1])
        beta[t] /= max(scales[t + 1], 1e-12)

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi01 = 0.0
    xi10 = 0.0
    denom_t = 0.0
    denom_f = 0.0
    for t in range(t_len - 1):
        trans = alpha[t][:, None] * transition * (emission[:, seq[t + 1]] * beta[t + 1])[None, :]
        trans /= max(trans.sum(), 1e-12)
        xi01 += trans[0, 1]
        xi10 += trans[1, 0]
        denom_t += gamma[t, 0]
        denom_f += gamma[t, 1]

    denom_g = gamma[:, 0].sum()
    num_g = gamma[:, 0][seq == 1].sum()
    denom_s = gamma[:, 1].sum()
    num_s = gamma[:, 1][seq == 0].sum()
    loglike = np.log(np.maximum(scales, 1e-12)).sum()

    return {
        "gamma": gamma,
        "xi01": xi01,
        "xi10": xi10,
        "denom_t": denom_t,
        "denom_f": denom_f,
        "num_g": num_g,
        "denom_g": denom_g,
        "num_s": num_s,
        "denom_s": denom_s,
        "loglike": loglike,
    }


def fit_bkt_skill(
    sequences: Sequence[Sequence[int]] | Sequence[int],
    n_restarts: int = 8,
    seed: int = 42,
    max_iter: int = 50,
    tol: float = 1e-5,
    fixed_p_g: float | None = None,
    fixed_p_s: float | None = None,
    fixed_p_f: float | None = None,
) -> dict[str, float]:
    """Fit a 2-state BKT model to one sequence or a list of student sequences."""
    normalized = _normalize_sequences(sequences)
    rng = np.random.default_rng(seed)

    if not normalized:
        return {"p_l0": 0.2, "p_t": 0.1, "p_g": 0.2, "p_s": 0.2, "p_f": 0.0}

    best_params: np.ndarray | None = None
    best_ll = -np.inf

    for _ in range(max(1, n_restarts)):
        first_observations = [seq[0] for seq in normalized if len(seq) > 0]
        init_l0 = float(np.clip(np.mean(first_observations) if first_observations else 0.2, 0.05, 0.95))
        params = np.array(
            [
                init_l0,
                rng.uniform(0.01, 0.30),
                fixed_p_g if fixed_p_g is not None else rng.uniform(0.05, 0.35),
                fixed_p_s if fixed_p_s is not None else rng.uniform(0.05, 0.35),
                fixed_p_f if fixed_p_f is not None else rng.uniform(0.00, 0.20),
            ],
            dtype=float,
        )

        last_ll = -np.inf
        for _ in range(max_iter):
            stats = {
                "l0_mastered": 0.0,
                "seq_count": 0.0,
                "xi01": 0.0,
                "xi10": 0.0,
                "denom_t": 0.0,
                "denom_f": 0.0,
                "num_g": 0.0,
                "denom_g": 0.0,
                "num_s": 0.0,
                "denom_s": 0.0,
            }
            ll = 0.0

            for seq in normalized:
                fb = _forward_backward(seq, params)
                if not fb:
                    continue
                ll += float(fb["loglike"])
                stats["l0_mastered"] += float(fb["gamma"][0, 1])
                stats["seq_count"] += 1.0
                stats["xi01"] += float(fb["xi01"])
                stats["xi10"] += float(fb["xi10"])
                stats["denom_t"] += float(fb["denom_t"])
                stats["denom_f"] += float(fb["denom_f"])
                stats["num_g"] += float(fb["num_g"])
                stats["denom_g"] += float(fb["denom_g"])
                stats["num_s"] += float(fb["num_s"])
                stats["denom_s"] += float(fb["denom_s"])

            new_params = np.array(
                [
                    stats["l0_mastered"] / max(stats["seq_count"], 1e-12),
                    params[1] if fixed_p_f is not None else stats["xi01"] / max(stats["denom_t"], 1e-12),
                    params[2] if fixed_p_g is not None else stats["num_g"] / max(stats["denom_g"], 1e-12),
                    params[3] if fixed_p_s is not None else stats["num_s"] / max(stats["denom_s"], 1e-12),
                    params[4] if fixed_p_f is not None else stats["xi10"] / max(stats["denom_f"], 1e-12),
                ],
                dtype=float,
            )
            new_params = np.clip(new_params, 1e-4, 1 - 1e-4)
            if fixed_p_g is not None:
                new_params[2] = fixed_p_g
            if fixed_p_s is not None:
                new_params[3] = fixed_p_s
            if fixed_p_f is not None:
                new_params[4] = fixed_p_f

            params = new_params
            if abs(ll - last_ll) < tol:
                break
            last_ll = ll

        if ll > best_ll:
            best_ll = ll
            best_params = params.copy()

    assert best_params is not None
    return {
        "p_l0": float(best_params[0]),
        "p_t": float(best_params[1]),
        "p_g": float(best_params[2]),
        "p_s": float(best_params[3]),
        "p_f": float(best_params[4]),
    }


fit_bkt_student = fit_bkt_skill


def bkt_predict_sequence(
    responses: Sequence[int],
    p_l0: float,
    p_t: float,
    p_g: float,
    p_s: float,
    p_f: float = 0.0,
) -> np.ndarray:
    """Return the probability of a correct response at each opportunity."""
    seq = np.asarray(responses, dtype=int)
    mastery = float(p_l0)
    preds: list[float] = []

    for obs in seq:
        pred = mastery * (1 - p_s) + (1 - mastery) * p_g
        preds.append(pred)
        if obs == 1:
            posterior = (mastery * (1 - p_s)) / max(pred, 1e-12)
        else:
            posterior = (mastery * p_s) / max(1 - pred, 1e-12)
        mastery = posterior * (1 - p_f) + (1 - posterior) * p_t

    return np.asarray(preds, dtype=float)


def get_prediction_column(predictions: pd.DataFrame) -> str:
    """Return the most likely prediction column name."""
    preferred = [
        "correct_predictions",
        "prediction",
        "predicted_probability",
        "prob_correct",
    ]
    for column in preferred:
        if column in predictions.columns:
            return column
    numeric_columns = [col for col in predictions.columns if pd.api.types.is_numeric_dtype(predictions[col])]
    if numeric_columns:
        return numeric_columns[-1]
    return predictions.columns[-1]


def predict_by_skill(
    dataframe: pd.DataFrame,
    skill_params: dict[str, dict[str, float]],
    *,
    skill_col: str = "skill_name",
    student_col: str = "anon_student_id",
    order_col: str = "opportunity",
    response_col: str = "correct",
    prediction_col: str = "correct_predictions",
) -> pd.DataFrame:
    """Predict each student-skill sequence and return a dataframe with predictions."""
    if dataframe.empty:
        out = dataframe.copy()
        out[prediction_col] = []
        return out

    parts: list[pd.DataFrame] = []
    for skill, skill_df in dataframe.groupby(skill_col, sort=False):
        params = skill_params[skill]
        skill_df = skill_df.sort_values([student_col, order_col]).copy()
        for _, student_df in skill_df.groupby(student_col, sort=False):
            student_df = student_df.sort_values(order_col).copy()
            student_df[prediction_col] = bkt_predict_sequence(
                student_df[response_col].to_numpy(),
                p_l0=params["p_l0"],
                p_t=params["p_t"],
                p_g=params["p_g"],
                p_s=params["p_s"],
                p_f=params.get("p_f", 0.0),
            )
            parts.append(student_df)

    return pd.concat(parts).sort_index()


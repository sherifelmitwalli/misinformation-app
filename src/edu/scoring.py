"""
Scoring and Evaluation Module for Tobacco Misinformation Credibility Assessment
Implementation of Algorithm 3 and performance validation metrics from:
Elmitwalli et al., "Development and validation of a multi-agent AI pipeline for automated 
credibility assessment of tobacco misinformation: a proof-of-concept study", 
Frontiers in Artificial Intelligence, 2025. doi: 10.3389/frai.2025.1659861
"""

from typing import Dict, List, Tuple
import numpy as np

# Component weights as validated in the published paper
WEIGHT_EVIDENCE_QUALITY = 0.40
WEIGHT_SCIENTIFIC_CONSENSUS = 0.25
WEIGHT_EVIDENCE_CONSISTENCY = 0.15
WEIGHT_EVIDENCE_RECENCY = 0.10
WEIGHT_SCIENTIFIC_PLAUSIBILITY = 0.10

# Five-level credibility scale
CATEGORIES = [
    ("Highly likely to be credible", 81.0, 100.0, 4),
    ("Likely to be credible", 61.0, 80.999, 3),
    ("Moderate credibility", 41.0, 60.999, 2),
    ("Unlikely to be credible", 21.0, 40.999, 1),
    ("Highly unlikely to be credible", 0.0, 20.999, 0)
]

def calculate_credibility_score(eq: float, sc: float, ec: float, er: float, sp: float) -> float:
    """
    Algorithm 3: Credibility scoring algorithm
    Score = (Eq * 0.40) + (Sc * 0.25) + (Ec * 0.15) + (Er * 0.10) + (Sp * 0.10)
    """
    for name, val in [("Eq", eq), ("Sc", sc), ("Ec", ec), ("Er", er), ("Sp", sp)]:
        if not (0.0 <= val <= 100.0):
            raise ValueError(f"Component {name} must be between 0 and 100, got {val}")

    score = (
        (eq * WEIGHT_EVIDENCE_QUALITY) +
        (sc * WEIGHT_SCIENTIFIC_CONSENSUS) +
        (ec * WEIGHT_EVIDENCE_CONSISTENCY) +
        (er * WEIGHT_EVIDENCE_RECENCY) +
        (sp * WEIGHT_SCIENTIFIC_PLAUSIBILITY)
    )
    return round(float(score), 2)

def classify_credibility_level(score: float) -> str:
    """Maps a 0-100 credibility score to the 5-level categorical framework."""
    if score >= 81.0:
        return "Highly likely to be credible"
    elif score >= 61.0:
        return "Likely to be credible"
    elif score >= 41.0:
        return "Moderate credibility"
    elif score >= 21.0:
        return "Unlikely to be credible"
    else:
        return "Highly unlikely to be credible"

def category_to_level_index(category_str: str) -> int:
    """Converts a category name into an ordinal level 0-4."""
    cat_lower = category_str.lower()
    if "highly likely" in cat_lower:
        return 4
    elif "likely" in cat_lower and "unlikely" not in cat_lower:
        return 3
    elif "moderate" in cat_lower:
        return 2
    elif "highly unlikely" in cat_lower:
        return 0
    elif "unlikely" in cat_lower:
        return 1
    raise ValueError(f"Unknown category string: {category_str}")

def compute_evaluation_metrics(auto_scores: List[float], manual_scores: List[float]) -> Dict[str, float]:
    """
    Computes framework performance metrics against expert benchmark:
    - Mean Absolute Error (MAE)
    - Mean Signed Difference (Auto - Manual bias)
    - Standard Deviation of Differences
    - Maximum Absolute Error
    - Exact-Level Category Agreement (%)
    - Adjacent-Level Category Agreement (%)
    - Quadratic-Weighted Cohen's Kappa (kappa)
    """
    if len(auto_scores) != len(manual_scores) or len(auto_scores) == 0:
        raise ValueError("Scores lists must be non-empty and of equal length.")

    n = len(auto_scores)
    diffs = np.array(auto_scores) - np.array(manual_scores)
    abs_diffs = np.abs(diffs)

    mae = float(np.mean(abs_diffs))
    mean_bias = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    max_error = float(np.max(abs_diffs))

    auto_levels = [category_to_level_index(classify_credibility_level(s)) for s in auto_scores]
    manual_levels = [category_to_level_index(classify_credibility_level(s)) for s in manual_scores]

    exact_matches = sum(1 for a, m in zip(auto_levels, manual_levels) if a == m)
    adjacent_matches = sum(1 for a, m in zip(auto_levels, manual_levels) if abs(a - m) <= 1)

    exact_agreement = (exact_matches / n) * 100.0
    adjacent_agreement = (adjacent_matches / n) * 100.0

    # Quadratic weighted Cohen's Kappa
    num_categories = 5
    conf_mat = np.zeros((num_categories, num_categories), dtype=float)
    for a, m in zip(auto_levels, manual_levels):
        conf_mat[m, a] += 1.0

    conf_mat /= n
    row_marginals = conf_mat.sum(axis=1)
    col_marginals = conf_mat.sum(axis=0)
    expected_mat = np.outer(row_marginals, col_marginals)

    weights = np.zeros((num_categories, num_categories), dtype=float)
    for i in range(num_categories):
        for j in range(num_categories):
            weights[i, j] = 1.0 - ((i - j) ** 2) / float((num_categories - 1) ** 2)

    po = np.sum(weights * conf_mat)
    pe = np.sum(weights * expected_mat)
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 1.0

    return {
        "MAE": round(mae, 2),
        "Mean_Signed_Difference": round(mean_bias, 2),
        "Std_Dev_Difference": round(std_diff, 2),
        "Max_Absolute_Error": round(max_error, 2),
        "Exact_Agreement_Percent": round(exact_agreement, 2),
        "Adjacent_Agreement_Percent": round(adjacent_agreement, 2),
        "Weighted_Cohen_Kappa": round(float(kappa), 2)
    }

import json

import numpy as np
from scipy import stats


def calculate_correlations(x, y):
    """
    Calculate both Pearson correlations with detailed steps

    Parameters:
    x, y: arrays of equal length containing paired observations

    Returns:
    dict: Detailed results of correlation calculations
    """

    def pearson_manual(x, y):
        "formula: r = Σ((x - μx)(y - μy)) / √(Σ(x - μx)² * Σ(y - μy)²)"

        # Step 1: Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Step 2: Calculate deviations
        x_dev = x - x_mean
        y_dev = y - y_mean

        # Step 3: Calculate sum of products of deviations
        sum_prod_dev = np.sum(x_dev * y_dev)

        # Step 4: Calculate sum of squared deviations
        sum_sq_dev_x = np.sum(x_dev**2)
        sum_sq_dev_y = np.sum(y_dev**2)

        # Step 5: Calculate correlation coefficient
        r = sum_prod_dev / np.sqrt(sum_sq_dev_x * sum_sq_dev_y)
        return r

    manual_res = pearson_manual(x, y)
    scipy_res = stats.pearsonr(x, y)[0]
    assert np.isclose(manual_res, scipy_res), "Manual calculation and SciPy verification do not match"

    return round(manual_res, 4)


method_list = [
    "textual_inversion_sd",
    "dreambooth_sd",
    "dreambooth_lora_sdxl",
    "blip_diffusion",
    "emu2",
    "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_g_sdxl",
]

print("------ Concept Preservation ------")
for method in method_list:
    human_rating_file1 = f"data_human_rating/merged_data/group1/{method}-cp.json"
    human_rating_file2 = f"data_human_rating/merged_data/group2/{method}-cp.json"
    gpt_rating_file = f"data_gpt_rating/concept_preservation_full/{method}.json"
    dino_rating_file = f"data_dino_rating/{method}.json"
    clip_rating_file = f"data_clipi_rating/{method}.json"

    def _get_rating(file):
        with open(file, "r") as f:
            data = json.load(f)
        sorted(data)
        rating = np.array(list(data.values()))
        return rating

    human_rating1 = _get_rating(human_rating_file1)
    human_rating2 = _get_rating(human_rating_file2)
    gpt_rating = _get_rating(gpt_rating_file)
    dino_rating = _get_rating(dino_rating_file)
    clip_rating = _get_rating(clip_rating_file)

    print(f"Method: {method}")

    results = calculate_correlations(human_rating1, human_rating2)
    print(f"Human rating1 vs Human rating2: {results}")

    results1 = calculate_correlations(human_rating1, gpt_rating)
    results2 = calculate_correlations(human_rating2, gpt_rating)
    print(f"Human rating vs GPT rating: {(results1+results2)/2:.3f}±{abs(results1-results2)/2:.3f}")

    results1 = calculate_correlations(human_rating1, dino_rating)
    results2 = calculate_correlations(human_rating2, dino_rating)
    print(f"Human rating vs DINO rating: {(results1+results2)/2:.3f}±{abs(results1-results2)/2:.3f}")

    results1 = calculate_correlations(human_rating1, clip_rating)
    results2 = calculate_correlations(human_rating2, clip_rating)
    print(f"Human rating vs CLIP rating: {(results1+results2)/2:.3f}±{abs(results1-results2)/2:.3f}")

    print()


print("------ Prompt Following ------")
for method in method_list:
    human_rating_file1 = f"data_human_rating/merged_data/group1/{method}-pf.json"
    human_rating_file2 = f"data_human_rating/merged_data/group2/{method}-pf.json"
    gpt_rating_file = f"data_gpt_rating/prompt_following_full/{method}.json"
    clip_rating_file = f"data_clipt_rating/{method}.json"

    def _get_rating(file):
        with open(file, "r") as f:
            data = json.load(f)
        sorted(data)
        rating = np.array(list(data.values()))
        return rating

    human_rating1 = _get_rating(human_rating_file1)
    human_rating2 = _get_rating(human_rating_file2)
    gpt_rating = _get_rating(gpt_rating_file)
    clip_rating = _get_rating(clip_rating_file)

    print(f"Method: {method}")

    results = calculate_correlations(human_rating1, human_rating2)
    print(f"Human rating1 vs Human rating2: {results}")

    results1 = calculate_correlations(human_rating1, gpt_rating)
    results2 = calculate_correlations(human_rating2, gpt_rating)
    print(f"Human rating vs GPT rating: {(results1+results2)/2:.3f}±{abs(results1-results2)/2:.3f}")

    results1 = calculate_correlations(human_rating1, clip_rating)
    results2 = calculate_correlations(human_rating2, clip_rating)
    print(f"Human rating vs CLIP rating: {(results1+results2)/2:.3f}±{abs(results1-results2)/2:.3f}")

    print()

import json
import os

import krippendorff
import numpy as np

from dreambench_plus.constants import GPT_RATING_DATA, HUMAN_RATING_MERGED_DATA, HUMAN_RATING_RAW_DATA, METHODS
from dreambench_plus.merge_data import merge_data_gpt_rating, merge_data_human_rating
from dreambench_plus.utils.misc import avg_std, is_nan, listdir


def kd_alpha(json_files):
    data = []
    for file in json_files:
        with open(file, "r") as f:
            data.append(json.load(f))

    keys = set()
    for d in data:
        keys = keys.union(set(d.keys()))

    reliability_data = [[] for _ in range(len(data))]
    for i, d in enumerate(data):
        for key in keys:
            score = d.get(key, np.nan)
            reliability_data[i].append(np.nan if is_nan(score) else score)

    return round(krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="interval"), 3)


if __name__ == "__main__":
    print("Step 1: Merge Human Rating Data ...")
    groups = listdir(HUMAN_RATING_RAW_DATA)
    for method in METHODS:
        for group in groups:
            merge_data_human_rating(
                dir=os.path.join(HUMAN_RATING_RAW_DATA, group),
                method=method,
                out_dir=os.path.join(HUMAN_RATING_MERGED_DATA, group),
            )

    print("Step 2: Calculate Krippendorff's Alpha of Human Rating Data ...")
    human_kd_alpha_cp = []
    human_kd_alpha_pf = []
    for method in METHODS:
        _method = method.replace(" ", "_").replace("-", "_").lower()
        json_files_cp = [os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-cp.json") for group in groups]
        json_files_pf = [os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-pf.json") for group in groups]

        human_kd_alpha_cp.append(kd_alpha(json_files_cp))
        human_kd_alpha_pf.append(kd_alpha(json_files_pf))

    print("Step 3: Merge GPT Rating Data ...")
    for _dir in listdir(GPT_RATING_DATA):
        for method in METHODS:
            _method = method.replace(" ", "_").replace("-", "_").lower()
            merge_data_gpt_rating(
                dir=os.path.join(GPT_RATING_DATA, _dir),
                method=_method,
            )

    print("Step 4: Calculate Krippendorff's Alpha of GPT Rating Data and Human Rating Data ...")
    ablation_settings_cp = [
        "full",
        "full2",
        "full3",
        "wo_internal_thinking",
        "w_human_prior",
        "w_cot",
        "wo_scoring_range",
        "wo_scoring_criteria",
        # additioanl ablation settings
        "gpt4v_full",
        "1shot_full",
        "1shot_w_cot",
        "2shot_full",
        "2shot_w_cot",
    ]
    gpt_kd_alpha_cp = {}
    for setting in ablation_settings_cp:
        setting_dir = os.path.join(GPT_RATING_DATA, f"concept_preservation_{setting}")
        gpt_kd_alpha_cp[setting] = []
        for method in METHODS:
            _method = method.replace(" ", "_").replace("-", "_").lower()
            json_files_cp = [os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-cp.json") for group in groups]
            gpt_kd_alpha_cp_with_human = [kd_alpha([json_file_cp, os.path.join(setting_dir, f"{_method}.json")]) for json_file_cp in json_files_cp]
            average_kd_alpha = round(sum(gpt_kd_alpha_cp_with_human) / len(json_files_cp), 3)
            gpt_kd_alpha_cp[setting].append(average_kd_alpha)

    ablation_settings_pf = [
        "full",
        "full2",
        "full3",
        "wo_internal_thinking",
        "wo_cot",
        "wo_scoring_range",
        "wo_scoring_criteria",
        # additioanl ablation settings
        "gpt4v_full",
    ]
    gpt_kd_alpha_pf = {}
    for setting in ablation_settings_pf:
        setting_dir = os.path.join(GPT_RATING_DATA, f"prompt_following_{setting}")

        gpt_kd_alpha_pf[setting] = []
        for method in METHODS:
            _method = method.replace(" ", "_").replace("-", "_").lower()
            json_files_pf = [os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-pf.json") for group in groups]
            gpt_kd_alpha_pf_with_human = [kd_alpha([json_file_pf, os.path.join(setting_dir, f"{_method}.json")]) for json_file_pf in json_files_pf]
            average_kd_alpha = round(sum(gpt_kd_alpha_pf_with_human) / len(json_files_pf), 3)
            gpt_kd_alpha_pf[setting].append(average_kd_alpha)

    print("------------ Concept Preservation ------------")
    print(f"{'Method'.ljust(32)} |  H-H  |     G-H     |")
    for i, method in enumerate(METHODS):
        avg, std = avg_std([gpt_kd_alpha_cp["full"][i], gpt_kd_alpha_cp["full2"][i], gpt_kd_alpha_cp["full3"][i]])
        print(f"{method.ljust(32)} | {human_kd_alpha_cp[i]:.3f} | {avg:.3f}±{std:.3f} |")
    print()

    print("------------ Ablation Study of CP ------------")
    for setting in ablation_settings_cp[3:]:
        print(f"------------ {setting} ------------")
        print(f"{'Method'.ljust(32)} |  G-H  | delta  |")
        for i, method in enumerate(METHODS):
            avg, std = avg_std([gpt_kd_alpha_cp["full"][i], gpt_kd_alpha_cp["full2"][i], gpt_kd_alpha_cp["full3"][i]])
            if gpt_kd_alpha_cp[setting][i] - avg > 0:
                print(f"{method.ljust(32)} | {gpt_kd_alpha_cp[setting][i]:.3f} | +{gpt_kd_alpha_cp[setting][i] - avg:.3f} |")
            else:
                print(f"{method.ljust(32)} | {gpt_kd_alpha_cp[setting][i]:.3f} | {gpt_kd_alpha_cp[setting][i] - avg:.3f} |")
        print()

    print("------------ Prompt Following ------------")
    print(f"{'Method'.ljust(32)} |  H-H  |     G-H     |")
    for i, method in enumerate(METHODS):
        avg, std = avg_std([gpt_kd_alpha_pf["full"][i], gpt_kd_alpha_pf["full2"][i], gpt_kd_alpha_pf["full3"][i]])
        print(f"{method.ljust(32)} | {human_kd_alpha_pf[i]:.3f} | {avg:.3f}±{std:.3f} |")
    print()

    print("------------ Ablation Study of PF ------------")
    for setting in ablation_settings_pf[3:]:
        print(f"------------ {setting} ------------")
        print(f"{'Method'.ljust(32)} |  G-H  | delta  |")
        for i, method in enumerate(METHODS):
            avg, std = avg_std([gpt_kd_alpha_pf["full"][i], gpt_kd_alpha_pf["full2"][i], gpt_kd_alpha_pf["full3"][i]])
            if gpt_kd_alpha_pf[setting][i] - avg >= 0:
                print(f"{method.ljust(32)} | {gpt_kd_alpha_pf[setting][i]:.3f} | +{gpt_kd_alpha_pf[setting][i] - avg:.3f} |")
            else:
                print(f"{method.ljust(32)} | {gpt_kd_alpha_pf[setting][i]:.3f} | {gpt_kd_alpha_pf[setting][i] - avg:.3f} |")
        print()

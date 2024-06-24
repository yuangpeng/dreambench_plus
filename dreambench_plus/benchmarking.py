import json
import os
from typing import Literal

from dreambench_plus.constants import GPT_RATING_DATA, HUMAN_RATING_MERGED_DATA, HUMAN_RATING_RAW_DATA, METHODS
from dreambench_plus.merge_data import merge_data_gpt_rating, merge_data_human_rating
from dreambench_plus.utils.misc import avg_std, listdir


def cal_score(file, filter: Literal["animal", "human", "object", "style", "photorealistic", "style_transfer", "imaginative"] | None = None):
    with open(file, "r") as f:
        data = json.load(f)
    if filter is not None:
        new_data = {}
        for k, v in data.items():
            if filter in k:
                new_data[k] = v
            elif filter == "photorealistic" and int(k.split("-")[-1]) >= 0 and int(k.split("-")[-1]) <= 3:
                new_data[k] = v
            elif filter == "style_transfer" and int(k.split("-")[-1]) >= 4 and int(k.split("-")[-1]) <= 6:
                new_data[k] = v
            elif filter == "imaginative" and int(k.split("-")[-1]) >= 7 and int(k.split("-")[-1]) <= 8:
                new_data[k] = v
        data = new_data

    return sum(data.values()) / len(data) / 4


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

    print("Step 2: Merge GPT Rating Data ...")
    for _dir in listdir(GPT_RATING_DATA):
        for method in METHODS:
            _method = method.replace(" ", "_").replace("-", "_").lower()
            merge_data_gpt_rating(
                dir=os.path.join(GPT_RATING_DATA, _dir),
                method=_method,
            )

    print(f"{'Method'.ljust(32)} | Human Score CP | GPT Score CP | Human Score PF | GPT Score PF |")
    for method in METHODS:
        _method = method.replace(" ", "_").replace("-", "_").lower()

        groups = listdir(HUMAN_RATING_MERGED_DATA)

        human_scores_cp = []
        for group in groups:
            human_score_cp_file = os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-cp.json")
            human_scores_cp.append(cal_score(human_score_cp_file))
        human_score_cp_avg, human_score_cp_std = avg_std(human_scores_cp)

        human_scores_pf = []
        for group in groups:
            human_score_pf_file = os.path.join(HUMAN_RATING_MERGED_DATA, group, f"{_method}-pf.json")
            with open(human_score_pf_file, "r") as f:
                human_score_pf = json.load(f)
            human_scores_pf.append(sum(human_score_pf.values()) / len(human_score_pf) / 4)
        human_score_pf_avg, human_score_pf_std = avg_std(human_scores_pf)

        multiple = ["full", "full2", "full3"]

        gpt_scores_cp = []
        gpt_scores_cp_animal = []
        gpt_scores_cp_human = []
        gpt_scores_cp_object = []
        gpt_scores_cp_style = []
        for m in multiple:
            gpt_score_cp_file = os.path.join(GPT_RATING_DATA, f"concept_preservation_{m}", f"{_method}.json")
            gpt_scores_cp.append(cal_score(gpt_score_cp_file))
            gpt_scores_cp_animal.append(cal_score(gpt_score_cp_file, "animal"))
            gpt_scores_cp_human.append(cal_score(gpt_score_cp_file, "human"))
            gpt_scores_cp_object.append(cal_score(gpt_score_cp_file, "object"))
            gpt_scores_cp_style.append(cal_score(gpt_score_cp_file, "style"))
        gpt_score_cp_avg, gpt_score_cp_std = avg_std(gpt_scores_cp)
        gpt_score_cp_animal_avg, gpt_score_cp_animal_std = avg_std(gpt_scores_cp_animal)
        gpt_score_cp_human_avg, gpt_score_cp_human_std = avg_std(gpt_scores_cp_human)
        gpt_score_cp_object_avg, gpt_score_cp_object_std = avg_std(gpt_scores_cp_object)
        gpt_score_cp_style_avg, gpt_score_cp_style_std = avg_std(gpt_scores_cp_style)

        gpt_scores_pf = []
        gpt_scores_pf_photorealistic = []
        gpt_scores_pf_style_transfer = []
        gpt_scores_pf_imaginative = []
        for m in multiple:
            gpt_score_pf_file = os.path.join(GPT_RATING_DATA, f"prompt_following_{m}", f"{_method}.json")
            gpt_scores_pf.append(cal_score(gpt_score_pf_file))
            gpt_scores_pf_photorealistic.append(cal_score(gpt_score_pf_file, "photorealistic"))
            gpt_scores_pf_style_transfer.append(cal_score(gpt_score_pf_file, "style_transfer"))
            gpt_scores_pf_imaginative.append(cal_score(gpt_score_pf_file, "imaginative"))
        gpt_score_pf_avg, gpt_score_pf_std = avg_std(gpt_scores_pf)
        gpt_score_pf_photorealistic_avg, gpt_score_pf_photorealistic_std = avg_std(gpt_scores_pf_photorealistic)
        gpt_score_pf_style_transfer_avg, gpt_score_pf_style_transfer_std = avg_std(gpt_scores_pf_style_transfer)
        gpt_score_pf_imaginative_avg, gpt_score_pf_imaginative_std = avg_std(gpt_scores_pf_imaginative)

        print(
            f"{method.ljust(32)} |  {human_score_cp_avg:.3f}±{human_score_cp_std:.3f}   | {gpt_score_cp_avg:.3f}±{gpt_score_cp_std:.3f}  |  {human_score_pf_avg:.3f}±{human_score_pf_std:.3f}   | {gpt_score_pf_avg:.3f}±{gpt_score_pf_std:.3f}  |"
        )

    print(f"{'      '.ljust(32)} |                GPT Score CP               |                       GPT Score PF                      |         |")
    print(f"{'Method'.ljust(32)} | Animal | Human | Object | Style | Overall | Photorealistic | Style Transfer | Imaginative | Overall | CP * PF |")
    for method in METHODS:
        _method = method.replace(" ", "_").replace("-", "_").lower()

        multiple = ["full", "full2", "full3"]

        gpt_scores_cp = []
        gpt_scores_cp_animal = []
        gpt_scores_cp_human = []
        gpt_scores_cp_object = []
        gpt_scores_cp_style = []
        for m in multiple:
            gpt_score_cp_file = os.path.join(GPT_RATING_DATA, f"concept_preservation_{m}", f"{_method}.json")
            gpt_scores_cp_animal.append(cal_score(gpt_score_cp_file, "animal"))
            gpt_scores_cp_human.append(cal_score(gpt_score_cp_file, "human"))
            gpt_scores_cp_object.append(cal_score(gpt_score_cp_file, "object"))
            gpt_scores_cp_style.append(cal_score(gpt_score_cp_file, "style"))
            gpt_scores_cp.append(cal_score(gpt_score_cp_file))
        gpt_score_cp_animal_avg, gpt_score_cp_animal_std = avg_std(gpt_scores_cp_animal)
        gpt_score_cp_human_avg, gpt_score_cp_human_std = avg_std(gpt_scores_cp_human)
        gpt_score_cp_object_avg, gpt_score_cp_object_std = avg_std(gpt_scores_cp_object)
        gpt_score_cp_style_avg, gpt_score_cp_style_std = avg_std(gpt_scores_cp_style)
        gpt_score_cp_avg, gpt_score_cp_std = avg_std(gpt_scores_cp)

        gpt_scores_pf = []
        gpt_scores_pf_photorealistic = []
        gpt_scores_pf_style_transfer = []
        gpt_scores_pf_imaginative = []
        for m in multiple:
            gpt_score_pf_file = os.path.join(GPT_RATING_DATA, f"prompt_following_{m}", f"{_method}.json")
            gpt_scores_pf_photorealistic.append(cal_score(gpt_score_pf_file, "photorealistic"))
            gpt_scores_pf_style_transfer.append(cal_score(gpt_score_pf_file, "style_transfer"))
            gpt_scores_pf_imaginative.append(cal_score(gpt_score_pf_file, "imaginative"))
            gpt_scores_pf.append(cal_score(gpt_score_pf_file))
        gpt_score_pf_photorealistic_avg, gpt_score_pf_photorealistic_std = avg_std(gpt_scores_pf_photorealistic)
        gpt_score_pf_style_transfer_avg, gpt_score_pf_style_transfer_std = avg_std(gpt_scores_pf_style_transfer)
        gpt_score_pf_imaginative_avg, gpt_score_pf_imaginative_std = avg_std(gpt_scores_pf_imaginative)
        gpt_score_pf_avg, gpt_score_pf_std = avg_std(gpt_scores_pf)

        print(
            f"{method.ljust(32)} | {gpt_score_cp_animal_avg:.3f}  | {gpt_score_cp_human_avg:.3f} | {gpt_score_cp_object_avg:.3f}  | {gpt_score_cp_style_avg:.3f} |  {gpt_score_cp_avg:.3f}  |      {gpt_score_pf_photorealistic_avg:.3f}     |     {gpt_score_pf_style_transfer_avg:.3f}      |    {gpt_score_pf_imaginative_avg:.3f}    |  {gpt_score_pf_avg:.3f}  |  {gpt_score_cp_avg * gpt_score_pf_avg:.3f}  |"
        )

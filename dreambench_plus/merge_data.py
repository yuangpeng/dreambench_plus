import glob
import json
import os


def merge_data_gpt_rating(dir, method):
    if dir[-1] == "/":
        dir = dir[:-1]

    method = method.replace(" ", "_").replace("-", "_").lower()
    files = glob.glob(f"{dir}/{method}/*.json")

    new_data = {}
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        collection_id = data["collection_id"]
        prompt_index = data["prompt_index"]
        score = data["score"]

        new_data[f"{collection_id}-{prompt_index}"] = score

    new_data = dict(sorted(new_data.items()))

    with open(f"{dir}/{method}.json", "w") as f:
        json.dump(new_data, f, indent=4)


def merge_data_human_rating(dir, method, out_dir):
    if dir[-1] == "/":
        dir = dir[:-1]

    if out_dir[-1] == "/":
        out_dir = out_dir[:-1]

    method = method.replace(" ", "_").replace("-", "_").lower()
    files = glob.glob(f"{dir}/*-{method}.json")

    new_data_cp = {}
    new_data_pf = {}
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        new_data_cp.update({k: v[0] for k, v in data.items()})
        new_data_pf.update({k: v[1] for k, v in data.items()})

    new_data_cp = dict(sorted(new_data_cp.items()))
    new_data_pf = dict(sorted(new_data_pf.items()))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/{method}-cp.json", "w") as f:
        json.dump(new_data_cp, f, indent=4)
    with open(f"{out_dir}/{method}-pf.json", "w") as f:
        json.dump(new_data_pf, f, indent=4)

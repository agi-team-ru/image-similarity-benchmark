import os
from typing import Dict, List, Tuple
import pandas as pd
from pprint import pprint
import re
import random

from dataset_params import ds_source_dir, ds_info_file, ds_key_score, ds_key_images

pairs_in_group_n_factor = 3


def main():
    file_paths: List[str] = []
    for f in os.listdir(ds_source_dir):
        fpath = os.path.join(ds_source_dir, f)
        file_paths.append(fpath)
    file_paths.sort()

    groups: Dict[str, List[str]] = {}
    group_extract_regex = re.compile(r"^(\w+)_\d+\.\w+$")
    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        match = group_extract_regex.match(base_name)
        if not match:
            raise Exception(f"Invalid name: {base_name}")
        group = match.group(1)
        groups.setdefault(group, []).append(file_path.removeprefix(ds_source_dir + "/"))

    true_pairs: List[Tuple[str, str]] = []
    false_pairs: List[Tuple[str, str]] = []

    for items in groups.values():
        pairs: List[Tuple[str, str]] = []
        n = len(items)
        for i in range(0, n):
            for j in range(i + 1, n):
                pairs.append((items[i], items[j]))
        true_pairs += random.sample(pairs, min(len(pairs), n * pairs_in_group_n_factor))

    for self_group, self_items in groups.items():
        other_group_items: List[str] = []
        for other_group, other_items in groups.items():
            if self_group == other_group:
                continue
            other_group_items += other_items
        if not other_group_items:
            continue
        for self_item in self_items:
            for other_item in random.sample(
                other_group_items,
                min(len(other_group_items), pairs_in_group_n_factor),
            ):
                false_pairs.append((self_item, other_item))

    df_img1: List[str] = []
    df_img2: List[str] = []
    df_score: List[float] = []

    for img1, img2 in true_pairs:
        df_img1.append(img1)
        df_img2.append(img2)
        df_score.append(1.0)

    for img1, img2 in false_pairs:
        df_img1.append(img1)
        df_img2.append(img2)
        df_score.append(0.0)

    df = pd.DataFrame(
        {ds_key_images[0]: df_img1, ds_key_images[1]: df_img2, ds_key_score: df_score}
    )

    df.sample(frac=1).to_csv(ds_info_file, index=False)

    print(f"Unique images: {len(file_paths)}")
    print(f"True similarity pairs: {len(true_pairs)}")
    print(f"False similarity pairs: {len(false_pairs)}")


if __name__ == "__main__":
    main()

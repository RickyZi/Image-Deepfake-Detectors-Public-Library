# import os

# dataset_path = '/dataset-disk/tb_dataset/tf2k_lr/social/facebook/'


# subfolder_counts= {
#     "real": {
#       "FFHQ": 500,
#       "FORLAB": 500
#     },
#     "fake": {
#       "StyleGAN/images-psi-0.7": 38,
#       "StableDiffusion1.5/general": 39,
#       "StableDiffusion1.5/faces": 39,
#       "StableDiffusion3/general": 38,
#       "StyleGAN3/conf-t-psi-0.5": 38,
#       "StableDiffusionXL/animals": 38,
#       "FLUX.1/faces": 39,
#       "FLUX.1/landscapes": 39,
#       "FLUX.1/general": 39,
#       "StableDiffusion2/animals": 39,
#       "StableDiffusion2/general": 39,
#       "StableDiffusionXL/landscapes": 38,
#       "StyleGAN3/conf-t-psi-0.7": 38,
#       "StableDiffusionXL/faces": 38,
#       "StableDiffusion2/faces": 39,
#       "StyleGAN2/conf-f-psi-1": 38,
#       "StableDiffusion3/faces": 38,
#       "StableDiffusion3/landscapes": 38,
#       "StyleGAN/images-psi-0.5": 38,
#       "StableDiffusionXL/general": 38,
#       "StableDiffusion2/landscapes": 39,
#       "StableDiffusion1.5/animals": 39,
#       "FLUX.1/animals": 39,
#       "StyleGAN2/conf-f-psi-0.5": 38,
#       "StableDiffusion3/animals": 38,
#       "StableDiffusion1.5/landscapes": 39
#     }
# }

# for root, dir, files in os.walk(dataset_path):
#     # print(f'root: {root}')
#     if files:
#         rel_dir = os.path.realpath(root)
#         print(f'rel_path {rel_dir} - len(file) {len(files)}')
#     # print()

import os
from collections import defaultdict

dataset_path = '/dataset-disk/tb_dataset/tf2k_lr/social/facebook/'
print(dataset_path)

subfolder_counts = {
    "real": {
        "FFHQ": 500,
        "FORLAB": 500
    },
    "fake": {
        "StyleGAN/images-psi-0.7": 38,
        "StableDiffusion1.5/general": 39,
        "StableDiffusion1.5/faces": 39,
        "StableDiffusion3/general": 38,
        "StyleGAN3/conf-t-psi-0.5": 38,
        "StableDiffusionXL/animals": 38,
        "FLUX.1/faces": 39,
        "FLUX.1/landscapes": 39,
        "FLUX.1/general": 39,
        "StableDiffusion2/animals": 39,
        "StableDiffusion2/general": 39,
        "StableDiffusionXL/landscapes": 38,
        "StyleGAN3/conf-t-psi-0.7": 38,
        "StableDiffusionXL/faces": 38,
        "StableDiffusion2/faces": 39,
        "StyleGAN2/conf-f-psi-1": 38,
        "StableDiffusion3/faces": 38,
        "StableDiffusion3/landscapes": 38,
        "StyleGAN/images-psi-0.5": 38,
        "StableDiffusionXL/general": 38,
        "StableDiffusion2/landscapes": 39,
        "StableDiffusion1.5/animals": 39,
        "FLUX.1/animals": 39,
        "StyleGAN2/conf-f-psi-0.5": 38,
        "StableDiffusion3/animals": 38,
        "StableDiffusion1.5/landscapes": 39
    }
}

REAL_FAKE_NAMES = {"real", "fake"}


def classify_path(root, base_path):
    """
    Given a directory `root` under `base_path`, figure out:
      - group_key: everything above the real/fake folder, flattened with '-'
                   (e.g. 'seasons/spring-SP01' -> 'seasons-spring-SP01'), or
                   '(root)' if real/fake sits directly under base_path.
      - real_fake: 'real' or 'fake'
      - algo_key:  everything below real/fake, joined with '/'
                   (e.g. 'StyleGAN/images-psi-0.7', or just 'FFHQ')

    Returns None if this directory isn't inside a real/fake branch
    (e.g. it's an intermediate folder with no files, or malformed).
    """
    rel = os.path.relpath(root, base_path)
    if rel == ".":
        return None
    parts = rel.split(os.sep)

    idx = next((i for i, p in enumerate(parts) if p.lower() in REAL_FAKE_NAMES), None)
    if idx is None:
        return None

    algo_sub_parts = parts[idx + 1:]
    if not algo_sub_parts:
        # files sitting directly in the real/fake folder, no algorithm subfolder
        return None

    group_parts = parts[:idx]
    group_key = "-".join(group_parts) if group_parts else "(root)"
    real_fake = parts[idx].lower()
    algo_key = "/".join(algo_sub_parts)
    return group_key, real_fake, algo_key


# results[group_key][real_fake][algo_key] = image count
results = defaultdict(lambda: defaultdict(dict))

for root, dirs, files in os.walk(dataset_path):
    if not files:
        continue
    classified = classify_path(root, dataset_path)
    if classified is None:
        continue
    group_key, real_fake, algo_key = classified
    results[group_key][real_fake][algo_key] = results[group_key][real_fake].get(algo_key, 0) + len(files)


# ---- Print summary ----
total_mismatches = 0
total_groups_checked = 0

for group_key in sorted(results.keys()):
    print(f"\n{group_key}")
    for real_fake in ["real", "fake"]:
        if real_fake not in results[group_key]:
            continue
        print(real_fake)

        expected = subfolder_counts.get(real_fake, {})
        found_algos = results[group_key][real_fake]
        all_algo_keys = sorted(set(expected.keys()) | set(found_algos.keys()))

        for algo_key in all_algo_keys:
            expected_count = expected.get(algo_key)
            found_count = found_algos.get(algo_key, 0)
            total_groups_checked += 1

            if algo_key not in found_algos:
                status = "MISSING FOLDER"
                total_mismatches += 1
            elif expected_count is None:
                status = "UNEXPECTED (not in expected list)"
                total_mismatches += 1
            elif found_count == expected_count:
                status = "OK"
            else:
                status = f"MISMATCH (expected {expected_count})"
                total_mismatches += 1

            print(f"  * {algo_key} -> {found_count} images  [{status}]")

print(f"\n=== Summary: {total_mismatches} mismatch(es) out of {total_groups_checked} checked (algorithm, real/fake, group) combinations ===")

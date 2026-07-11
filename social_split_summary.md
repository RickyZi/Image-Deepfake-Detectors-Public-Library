# Split Summary Report — Facebook

Generated from `full_social_split.json` (social = **Facebook**) and `tf2k_SOCIAL_splits.json`.

## 1. Matching Check Between the Two Split Files

- Total samples in `full_social_split.json`: **2000**
- Total entries in `tf2k_SOCIAL_splits.json` (train+val+test): **2000** (train=1210, val=408, test=382)
- Entries only in `full_social_split.json`: **0**
- Entries only in `tf2k_SOCIAL_splits.json`: **0**
- Overlap between train/val: **0**, train/test: **0**, val/test: **0**
- Samples that could not be assigned to any split: **0**

**Result: ✅ PASS — files match perfectly**

## 2. Expected Numerosity Check

- Declared totals in `full_social_split.json`: n_real=1000, n_fake=1000 (total=2000)
- Sum of train+val+test across all families: **2000**
- ✅ All family totals (train+val+test) match `subfolder_counts` exactly.

## 3. Per-Family Counts (fine-grained)

| Label | Family | Train | Val | Test | Total |
|---|---|---|---|---|---|
| Fake | FLUX.1/animals | 24 | 8 | 7 | 39 |
| Fake | FLUX.1/faces | 24 | 8 | 7 | 39 |
| Fake | FLUX.1/general | 24 | 8 | 7 | 39 |
| Fake | FLUX.1/landscapes | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion1.5/animals | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion1.5/faces | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion1.5/general | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion1.5/landscapes | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion2/animals | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion2/faces | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion2/general | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion2/landscapes | 24 | 8 | 7 | 39 |
| Fake | StableDiffusion3/animals | 23 | 8 | 7 | 38 |
| Fake | StableDiffusion3/faces | 23 | 8 | 7 | 38 |
| Fake | StableDiffusion3/general | 23 | 8 | 7 | 38 |
| Fake | StableDiffusion3/landscapes | 23 | 8 | 7 | 38 |
| Fake | StableDiffusionXL/animals | 23 | 8 | 7 | 38 |
| Fake | StableDiffusionXL/faces | 23 | 8 | 7 | 38 |
| Fake | StableDiffusionXL/general | 23 | 8 | 7 | 38 |
| Fake | StableDiffusionXL/landscapes | 23 | 8 | 7 | 38 |
| Fake | StyleGAN/images-psi-0.5 | 23 | 8 | 7 | 38 |
| Fake | StyleGAN/images-psi-0.7 | 23 | 8 | 7 | 38 |
| Fake | StyleGAN2/conf-f-psi-0.5 | 23 | 8 | 7 | 38 |
| Fake | StyleGAN2/conf-f-psi-1 | 23 | 8 | 7 | 38 |
| Fake | StyleGAN3/conf-t-psi-0.5 | 23 | 8 | 7 | 38 |
| Fake | StyleGAN3/conf-t-psi-0.7 | 23 | 8 | 7 | 38 |
| Real | FFHQ | 300 | 100 | 100 | 500 |
| Real | FORLAB | 300 | 100 | 100 | 500 |
| **GRAND TOTAL** | | **1210** | **408** | **382** | **2000** |

## 4. Per-Algorithm Counts (aggregated across variants)

| Label | Algo | Train | Val | Test | Total |
|---|---|---|---|---|---|
| Fake | FLUX.1 | 96 | 32 | 28 | 156 |
| Fake | StableDiffusion1.5 | 96 | 32 | 28 | 156 |
| Fake | StableDiffusion2 | 96 | 32 | 28 | 156 |
| Fake | StableDiffusion3 | 92 | 32 | 28 | 152 |
| Fake | StableDiffusionXL | 92 | 32 | 28 | 152 |
| Fake | StyleGAN | 46 | 16 | 14 | 76 |
| Fake | StyleGAN2 | 46 | 16 | 14 | 76 |
| Fake | StyleGAN3 | 46 | 16 | 14 | 76 |
| Real | FFHQ | 300 | 100 | 100 | 500 |
| Real | FORLAB | 300 | 100 | 100 | 500 |

## 5. Split Ratio Overview

| Split | Samples | Percentage of Total |
|---|---|---|
| Train | 1210 | 60.5% |
| Val | 408 | 20.4% |
| Test | 382 | 19.1% |
| **Total** | **2000** | **100.0%** |

_Note: individual small-count fake families cannot always hit an exact 60/20/20 split due to integer rounding; check the aggregated totals above for the overall split balance._

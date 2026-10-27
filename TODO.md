# TODO

Clean up repo to prepare it for possibile publication

- clean up code
- remove old comments
- add citation to original TB repo
  - something like: "This repo containes a customized verison of the code from TrueBees' Image Deepfake Detectors repo. The best performing models from their paper [cite paper] were selected and fine-tuned on the novel dataset of LR presets"
  - important! add ref to repo and paper
- describe code in the README
  - fine-tuning strategies
  - LoRA for CLIP (add references)
  - add refernces in the code
- add example usage
- add updated requirements -> uv?
- describe outputs and expected files
- describe modification for training and testing to make code run faster
  - tested on whole dataset and not just single split as in the TB paper
  - trn-val-tst splits
- collect bash script for training and testing the models (also for ensambles)
- add link to datasets
- add link to model weights
- add explaination on how social data were collected (no one-to-one match with presocial)
  - describe solution for resampling
  - new social split

Training strategy

- for each model updated the dataloader with custom version of the tf2k dataset (not interested in having results for each single partition but overall)
- added specific file for finetuning strategy
- added logger
- modified model config file to load the whole dataset and not just single algo splits.

----

## Pipeline demo

- input image
- check manifest
- c-mapper -> classify preset from actions
- return preset name
- load best model FT on that preset
- modify detect.py to output classification

-> classify multiple images from a folder.

## Generalization Experiemnts

what happens if prest wrongly classified as TM01 on SM01 imgs?

test models in non-aligned setting -> update way to load model to handle this case.

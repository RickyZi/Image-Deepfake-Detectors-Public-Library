#!/usr/bin/env bash

# missing baseline 

# R50_TF
python3 launcher.py --detector R50_TF --phases test  --dataset auto+/retro-AR01
python3 launcher.py --detector R50_TF --phases test  --dataset auto+/retro-AR09

# R50_nodown
python3 launcher.py --detector R50_nodown --phases test  --dataset auto+/retro-AR01
python3 launcher.py --detector R50_nodown --phases test  --dataset auto+/retro-AR09

# CLIP-D
python3 launcher.py --detector CLIP-D --phases test  --dataset auto+/retro-AR01
python3 launcher.py --detector CLIP-D --phases test  --dataset auto+/retro-AR09
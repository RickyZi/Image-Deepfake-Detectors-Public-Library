#!/usr/bin/env bash
#R50_nodown
# python3 utils/plot_tables_singlerun.py --unfreezeL4 --social facebook
python3 utils/plot_tables_singlerun.py --unfreezeL4 --social telegram
python3 utils/plot_tables_singlerun.py --unfreezeL4 --social twitter

#R50_TF
python3 utils/plot_tables_singlerun.py --unfreezeL4 --social facebook --model R50_TF
python3 utils/plot_tables_singlerun.py --unfreezeL4 --social telegram --model R50_TF
python3 utils/plot_tables_singlerun.py --unfreezeL4 --social twitter --model R50_TF

# CLIP-D
python3 utils/plot_tables_singlerun.py --lora --social facebook --model R50_TF
python3 utils/plot_tables_singlerun.py --lora --social telegram --model R50_TF
python3 utils/plot_tables_singlerun.py --lora --social twitter --model R50_TF

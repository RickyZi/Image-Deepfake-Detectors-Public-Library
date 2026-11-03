# ------ #
# CLIP-D #
# ------ #
# season
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn_TM01
python3 launcher.py --detector R50_TF --phases train --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 # unfreeze L4 when FT
python3 launcher.py --detector R50_nodown --phases train --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# style
# bw_BW01  cinematic2_CN11  cinematic_CN01  film_inspired_boldbw  film_inspired_coolbw  film_inspired_warmgold  futuristic_FT01  vintage_VN01
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw_BW01
python3 launcher.py --detector R50_TF --phases both --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
python3 launcher.py --detector R50_nodown --phases both --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
# # adaptive
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_strong
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg_strong --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/blurbg_strong --ft --r50unfreezeL4
# # subject
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape_LN01

# ----------------- #



# python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape_LN01 --ft --r50unfreezeL4

# ----------------- #



# python3 launcher.py --detector R50_nodown --phases both --dataset subject/landscape_LN01 --ft --r50unfreezeL4
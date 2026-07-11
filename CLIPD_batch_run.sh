#!/usr/bin/env bash
# run multiple demo in batch, e.g. for different presets


python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky_bluedrama --resume
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject_pop

# ---------------------------- #
# test baseline #
# ---------------------------- #

# # CLIP-D
# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/spring_SP01

# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/bw_BW01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_warmgold

# python3 launcher.py --detector CLIP-D --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/vintage_VN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_coolbw

# python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_boldbw

# # missin pretrained on adaptive and subject (just test the pretrained model on those)
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/blurbg_subtle 
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/blurbg_strong
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/enhance_portrait 
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/sky_bluedrama
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/subject_pop

# # subject
# # landscape_LN01  travel2_TR11  travel_TR01
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/landscape_LN01
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/travel2_TR11
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/travel_TR01 


# --------- # 
# FT model  #
# --------- #

# CLIP-D -> already has frozen bb and only classification head active
# circa 25 epochs on avg (between 20 and 30)

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/autumn_TM01 --ft # started trn-tst

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/spring_SP01 --ft # check (launched training 9/6 9:50)

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/summer_SM01 --ft # check (launched training 9/6 9:50)

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/winter_WN01 --ft

# python3 launcher.py --detector CLIP-D --phases both --dataset style/bw_BW01 --ft

# python3 launcher.py --detector CLIP-D --phases both --dataset style/cinematic_CN01 --ft # only one epoch? (still running)

# python3 launcher.py --detector CLIP-D --phases both --dataset style/cinematic2_CN11 --ft

# python3 launcher.py --detector CLIP-D --phases both --dataset style/filminspired_warmgold --ft # 

# python3 launcher.py --detector CLIP-D --phases both --dataset style/futuristic_FT01 --ft
 
# python3 launcher.py --detector CLIP-D --phases both --dataset style/vintage_VN01 --ft 

# ----------------- #
# add & FT MLP head #
# ----------------- #
# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/autumn_TM01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/spring_SP01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/summer_SM01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset seasons/winter_WN01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset style/bw_BW01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset style/cinematic_CN01 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset style/cinematic2_CN11 --ft --mlp

# python3 launcher.py --detector CLIP-D --phases both --dataset style/filminspired_warmgold --ft --mlp 

# python3 launcher.py --detector CLIP-D --phases both --dataset style/futuristic_FT01 --ft --mlp
 
# python3 launcher.py --detector CLIP-D --phases both --dataset style/vintage_VN01 --ft --mlp

# CLIP-D LoRA
# seasons
# autumn_TM01  spring_SP01  summer_SM01  winter_WN01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn_TM01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/spring_SP01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/summer_SM01 
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/winter_WN01

# style
# bw_BW01  cinematic2_CN11  cinematic_CN01  film_inspired_boldbw  film_inspired_coolbw  film_inspired_warmgold  futuristic_FT01  vintage_VN01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw_BW01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic_CN01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_warmgold
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_coolbw
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_boldbw 
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/futuristic_FT01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/vintage_VN01

# adaptive
# blurbg_subtle  blurbg_strong  enhance_portrait  sky_bluedrama  subject_pop
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_subtle # failed due to wrong dataset name!!! REDO!!!!
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_strong
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/enhance_portrait # here
## somehow skipped

# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_subtle

# # subject
# # landscape_LN01  travel2_TR11  travel_TR01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape_LN01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2_TR11
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel_TR01 



#!/usr/bin/env bash
# run multiple demo in batch, e.g. for different presets


# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky_bluedrama --resume
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject_pop

# ---------------------------- #
# test baseline #
# ---------------------------- #

# # CLIP-D
python3 launcher.py --detector CLIP-D --phases test --dataset seasons/autumn_TM01

python3 launcher.py --detector CLIP-D --phases test --dataset seasons/spring_SP01

python3 launcher.py --detector CLIP-D --phases test --dataset seasons/summer_SM01

python3 launcher.py --detector CLIP-D --phases test --dataset seasons/winter_WN01

python3 launcher.py --detector CLIP-D --phases test --dataset style/bw_BW01

python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic_CN01

python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic2_CN11

python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_warmgold

python3 launcher.py --detector CLIP-D --phases test --dataset style/futuristic_FT01

python3 launcher.py --detector CLIP-D --phases test --dataset style/vintage_VN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_coolbw

# python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_boldbw

# adaptive
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/blurbg_subtle 
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/blurbg_strong
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/enhance_portrait 
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/sky_bluedrama
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset adaptive/subject_pop

# # # subject
# # # landscape_LN01  travel2_TR11  travel_TR01
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/landscape_LN01
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/travel2_TR11
# python launcher.py --detector CLIP-D --phases test  --tf2k True --dataset subject/travel_TR01 

## --------------------------------------------- ##
## TEST BASELINES ##
# New presets
# portraits/
# bw-PB01/              deepskin-PD01-red/    edgy-PE01/            group-PG06/           light-skin-PL10/      medium-skin-PM11/
# deepskin-PD01-orange/ deepskin-PD01-yellow/ group-PG01/           group-PG07/           lightskin-PL01/       mediumskin-PM01/

# python launcher.py  --detector CLIP-D --phases test   --dataset portraits/bw-PB01
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/edgy-PE01
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/lightskin-PL01
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/group-PG01
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/mediumskin-PM01
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/deepskin-PD01-red   
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/deepskin-PD01-orange
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/deepskin-PD01-yellow

python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/group-PG06
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/group-PG07
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/light-skin-PL10
python3 launcher.py --detector CLIP-D --phases test   --dataset portraits/medium-skin-PM11

# subject/
# concerts-CT01/           concerts-CT10/           food-FD04/               lifestyle-LF01/          travel2_TR11/            urban-architecture-UA08/
# concerts-CT09/           food-FD01/               landscape_LN01/          lifestyle-LF08/          travel_TR01/             urbanarchitecture-UA01/     

python3 launcher.py --detector CLIP-D --phases test   --dataset subject/concerts-CT01
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/concerts-CT09
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/concerts-CT10
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/food-FD01
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/food-FD04
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/lifestyle-LF01
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/lifestyle-LF08
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/urbanarchitecture-UA01
python3 launcher.py --detector CLIP-D --phases test   --dataset subject/urban-architecture-UA08

# video/
#  creative-VCR01/ creative-VCR01/ creative-VCR08/ creative-VCR12/
python3 launcher.py --detector CLIP-D --phases test   --dataset video/creative-VCR01
python3 launcher.py --detector CLIP-D --phases test   --dataset video/creative-VCR08
python3 launcher.py --detector CLIP-D --phases test   --dataset video/creative-VCR12

# auto +
python3 launcher.py --detector CLIP-D --phases test  --dataset auto+/retro-AR01
python3 launcher.py --detector CLIP-D --phases test  --dataset auto+/retro-AR09


# --------- # 
# FT model  #
# --------- #
# CLIP-D LoRA
# seasons
# autumn_TM01  spring_SP01  summer_SM01  winter_WN01
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn_TM01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/spring_SP01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/summer_SM01 
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/winter_WN01

# style
# bw_BW01  cinematic2_CN11  cinematic_CN01  film_inspired_boldbw  film_inspired_coolbw  film_inspired_warmgold  futuristic_FT01  vintage_VN01
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw_BW01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic_CN01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_warmgold
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_coolbw
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film_inspired_boldbw 
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/futuristic_FT01
# python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset style/vintage_VN01

# adaptive
# blurbg_subtle  blurbg_strong  enhance_portrait  sky_bluedrama  subject_pop
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_subtle # failed due to wrong dataset name!!! REDO!!!!
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg_strong
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/enhance_portrait # here
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky_bluedrama
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject_pop
## somehow skipped

# # # subject
# # # landscape_LN01  travel2_TR11  travel_TR01
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape_LN01
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2_TR11
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel_TR01 

# # subject
# # landscape_LN01  travel2_TR11  travel_TR01
python launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape_LN01
python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2_TR11
# python launcher.py --detector CLIP-D --phases train  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel_TR01 


## --------------------------------------------- ##
## CLIP-D FT ##
# # auto+/ retro-AR01/ retro-AR09/
# python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset auto+/retro-AR01
# python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset auto+/retro-AR09

# portraits/
# bw-PB01/              deepskin-PD01-red/    edgy-PE01/            group-PG06/           light-skin-PL10/      medium-skin-PM11/
# deepskin-PD01-orange/ deepskin-PD01-yellow/ group-PG01/           group-PG07/           lightskin-PL01/       mediumskin-PM01/

# # python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/bw-PB01
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/edgy-PE01
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/lightskin-PL01
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/group-PG01
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/mediumskin-PM01
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/deepskin-PD01-red   
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/deepskin-PD01-orange
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/deepskin-PD01-yellow

# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/group-PG06
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/group-PG07
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/light-skin-PL10
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset portraits/medium-skin-PM11

# # subject/
# # concerts-CT01/           concerts-CT10/           food-FD04/               lifestyle-LF01/          travel2_TR11/            urban-architecture-UA08/
# # concerts-CT09/           food-FD01/               landscape_LN01/          lifestyle-LF08/          travel_TR01/             urbanarchitecture-UA01/     

# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/concerts-CT01
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/concerts-CT09
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/concerts-CT10
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/food-FD01
# # python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/food-FD04
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/lifestyle-LF01
# # python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/lifestyle-LF08
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/urbanarchitecture-UA01
# # python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/urban-architecture-UA08

# # video/
# #  creative-VCR01/ creative-VCR01/ creative-VCR08/ creative-VCR12/
# python3 launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset video/creative-VCR01
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset video/creative-VCR08
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset video/creative-VCR12


# # # auto+/ retro-AR01/ retro-AR09/
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset auto+/retro-AR01
# # python3 launcher.py --detector CLIP-D --phases both  --weights-name lora_r4_qv --tf2k True --ft --dataset auto+/retro-AR09
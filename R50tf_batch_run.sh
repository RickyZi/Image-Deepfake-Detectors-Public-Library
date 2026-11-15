#!/usr/bin/env bash


# ------------------------------------------ #
# ---------------- BASELINE ---------------- #
# ------------------------------------------ #
# # R50_TF
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector R50_TF --phases test --dataset style/bw_BW01

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_warmgold

# python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage_VN01

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_coolbw

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_boldbw

# # # missin pretrained on adaptive and subject (just test the pretrained model on those)
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/blurbg_subtle 
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/blurbg_strong
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/enhance_portrait 
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/sky_bluedrama
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/subject_pop

# # # subject
# # # landscape_LN01  travel2_TR11  travel_TR01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/landscape_LN01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/travel2_TR11
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/travel_TR01 

# ## --------------------------------------------- ##
# ## TEST BASELINES ##
# # New presets
# # portraits/
# # bw-PB01/              deepskin-PD01-red/    edgy-PE01/            group-PG06/           light-skin-PL10/      medium-skin-PM11/
# # deepskin-PD01-orange/ deepskin-PD01-yellow/ group-PG01/           group-PG07/           lightskin-PL01/       mediumskin-PM01/

# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/bw-PB01
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/edgy-PE01
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/lightskin-PL01
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/group-PG01
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/mediumskin-PM01
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/deepskin-PD01-red   
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/deepskin-PD01-orange
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/deepskin-PD01-yellow

# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/group-PG06
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/group-PG07
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/light-skin-PL10
# python3 launcher.py --detector R50_TF --phases test   --dataset portraits/medium-skin-PM11

# # subject/
# # concerts-CT01/           concerts-CT10/           food-FD04/               lifestyle-LF01/          travel2_TR11/            urban-architecture-UA08/
# # concerts-CT09/           food-FD01/               landscape_LN01/          lifestyle-LF08/          travel_TR01/             urbanarchitecture-UA01/     

# python3 launcher.py --detector R50_TF --phases test   --dataset subject/concerts-CT01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/concerts-CT09
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/concerts-CT10
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/food-FD01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/food-FD04
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/lifestyle-LF01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/lifestyle-LF08
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/urbanarchitecture-UA01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/urban-architecture-UA08

# # video/
# #  creative-VCR01/ creative-VCR01/ creative-VCR08/ creative-VCR12/
# python3 launcher.py --detector R50_TF --phases test   --dataset video/creative-VCR01
# python3 launcher.py --detector R50_TF --phases test   --dataset video/creative-VCR08
# python3 launcher.py --detector R50_TF --phases test   --dataset video/creative-VCR12

# auto+/ retro-AR01/ retro-AR09/
python3 launcher.py --detector R50_TF --phases test  --dataset auto+/retro-AR01
python3 launcher.py --detector R50_TF --phases test  --dataset auto+/retro-AR09


# ------------------------------------------ #
# --------------- FT model ----------------- #
# ------------------------------------------ #

# missing -> seasons, style 

# -------------------------------------------------- #

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn_TM01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter_WN01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/bw_BW01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2_CN11 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_warmgold --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset style/filminspired_warmgold --ft 
 
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic_FT01 --ft  
 
# # python3 launcher.py --detector R50_TF --phases both --dataset style/vintage_VN01 --ft 

# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage_VN01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_coolbw --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_boldbw --ft --r50unfreezeL4

# # missin pretrained on adaptive and subject (just test the pretrained model on those)
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/blurbg_strong --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/enhance_portrait --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset adaptive/subject_pop --ft --r50unfreezeL4

# # subject
# # landscape_LN01  travel2_TR11  travel_TR01
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/landscape_LN01 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/travel2_TR11 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases test   --dataset subject/travel_TR01 --ft --r50unfreezeL4



## --------------------------------------------- ##
## New presets:
# portraits/
# bw-PB01/              deepskin-PD01-red/    edgy-PE01/            group-PG06/           light-skin-PL10/      medium-skin-PM11/
# deepskin-PD01-orange/ deepskin-PD01-yellow/ group-PG01/           group-PG07/           lightskin-PL01/       mediumskin-PM01/

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/spring_SP01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/summer_SM01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/winter_WN01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases train --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic_CN01 --ft  --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic2_CN11 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset style/filminspired_warmgold --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic_FT01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases both --dataset style/vintage_VN01 --ft --r50unfreezeL4 # unfreeze L4 when FT
# python3 launcher.py --detector R50_TF --phases train --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset adaptive/enhance_portrait --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train  --dataset adaptive/subject_pop --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset style/film_inspired_boldbw --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset style/film_inspired_coolbw --ft --r50unfreezeL4
# # python3 launcher.py --detector R50_TF --phases train --dataset subject/landscape_LN01 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset subject/travel_TR01 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset subject/travel2_TR11 --ft --r50unfreezeL4

# subject/
# concerts-CT01/           concerts-CT10/           food-FD04/               lifestyle-LF01/          travel2_TR11/            urban-architecture-UA08/
# concerts-CT09/           food-FD01/               landscape_LN01/          lifestyle-LF08/          travel_TR01/             urbanarchitecture-UA01/     

# python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/concerts-CT01
# # python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/concerts-CT09
# # python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/concerts-CT10
# python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/food-FD01
# # python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/food-FD04
# python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/lifestyle-LF01
# # python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/lifestyle-LF08
# python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/urbanarchitecture-UA01
# # python3 launcher.py --detector R50_TF --phases test   --ft --r50unfreezeL4 --dataset subject/urban-architecture-UA08

# python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/enhance_portrait --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/subject_pop --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset style/film_inspired_boldbw --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset style/film_inspired_coolbw --ft --r50unfreezeL4
# # python3 launcher.py --detector R50_TF --phases train --dataset subject/landscape_LN01 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset subject/travel_TR01 --ft --r50unfreezeL4
# python3 launcher.py --detector R50_nodown --phases train --dataset subject/travel2_TR11 --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases test --trn-dataset seasons/autumn_TM01  --dataset seasons/spring_SP01 --ft --r50unfreezeL4

# datasets: seasons/autumn_TM01, seasons/spring_SP01, seasons/summer_SM01, seasons/winter_WN01, 
# style/bw_BW01, style/cinematic_CN01, style/cinematic2_CN11, style/filminspired_warmgold, style/futuristic_FT01, style/vintage_VN01 
# adaptive/blurbg_strong, adaptive/blurbg_subtle, adaptive/enhance_portrait, adaptive/sky_bluedrama, adaptive/subject_pop, 
# subject/landscape_LN01, subject/travel2_TR11, subject/travel_TR01

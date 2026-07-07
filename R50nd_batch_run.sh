#!/usr/bin/env bash
# run multiple demo in batch, e.g. for different presets

# #  season/SM01
# # echo "Running demo for season/SM01 preset..."
# python3 launcher.py --demo --demo-dataset season_SM01 --demo-detector P2G

# #  season/SP01
# echo "Running demo for season/SP01 preset..."
# python3 launcher.py --demo --demo-dataset season_SP01 --demo-detector P2G

# #  season/TM01
# echo "Running demo for season/TM01 preset..."
# python3 launcher.py --demo --demo-dataset season_TM01 --demo-detector P2G

# #  season/WN01
# echo "Running demo for season/WN01 preset..."
# python3 launcher.py --demo --demo-dataset season_WN01 --demo-detector P2G

# #  style/BW01
# echo "Running demo for style/BW01 preset..."
# python3 launcher.py --demo --demo-dataset style_BW01 --demo-detector P2G

# #  style/CN01
# echo "Running demo for style/CN01 preset..."
# python3 launcher.py --demo --demo-dataset style_CN01 --demo-detector P2G

# #  style/CN11
# echo "Running demo for style/CN11 preset..."
# python3 launcher.py --demo --demo-dataset style_CN11 --demo-detector P2G

# #  style/FT01
# echo "Running demo for style/FT01 preset..."
# python3 launcher.py --demo --demo-dataset style_FT01 --demo-detector P2G

# #  style/VN01
# echo "Running demo for style/VN01 preset..."
# python3 launcher.py --demo --demo-dataset style_VN01 --demo-detector P2G

# #  style/warmgold
# echo "Running demo for style/warmgold preset..."
# python3 launcher.py --demo --demo-dataset style_warmgold --demo-detector P2G


# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn_TM01

# ------------------------------------------ #
# ---------------- BASELINE ---------------- #
# ------------------------------------------ #
# # R50_nodown
# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector R50_nodown --phases test --dataset sedasons/spring_SP01

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector R50_nodown --phases test --dataset style/bw_BW01

# python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector R50_nodown --phases test --dataset style/filminspired_warmgold

# python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage_VN01


# ------------------------------------------ #
# --------------- FT model ----------------- #
# ------------------------------------------ #
# R50_nodown -> trained only classification head
# python3 launcher.py --detector R50_nodown --phases both --dataset seasons/autumn_TM01 --ft 

# python3 launcher.py --detector R50_nodown --phases both --dataset seasons/spring_SP01 --ft

# python3 launcher.py --detector R50_nodown --phases both --dataset seasons/summer_SM01 --ft

# python3 launcher.py --detector R50_nodown --phases both --dataset seasons/winter_WN01 --ft

# python3 launcher.py --detector R50_nodown --phases both --dataset style/bw_BW01 --ft # check

     # check just 1 epoch (style/cinematic_CN01)

# python3 launcher.py --detector R50_nodown --phases both --dataset style/cinematic2_CN11 --ft

# python3 launcher.py --detector R50_nodown --phases both --dataset style/filminspired_warmgold --ft # check
 
# # python3 launcher.py --detector R50_nodown --phases both --dataset style/futuristic_FT01 --ft  # check
 
# python3 launcher.py --detector R50_nodown --phases both --dataset style/vintage_VN01 --ft # check


# ------------------------------------------- #
# ---- Unfreezing layer 4 in R50 when FT ---- #
# ------------------------------------------- #
# ------------------------------------------------------------------- #
# facebook
# seasons
# autumn-TM01  spring-SP01  summer-SM01  winter-WN01
python3 launcher.py --detector R50_nodown --phases both --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset seasons/spring-SP01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social facebook

# style
# bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
python3 launcher.py --detector R50_nodown --phases both --dataset style/bw-BW01 --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/cinematic-CN01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/cinematic2-CN11 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social facebook # check this!!!
python3 launcher.py --detector R50_nodown --phases both --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social facebook

# adaptive 
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social facebook


# subject
# landscape-LN01  travel-TR01  travel2-TR11
python3 launcher.py --detector R50_nodown --phases both --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_nodown --phases both --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social facebook

# ------------------------------------------------------------------- #
# R50_TF #
# ------------------------------------------------------------------- #
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social facebook

# adaptive 
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social facebook


# subject
# landscape-LN01  travel-TR01  travel2-TR11
python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social facebook

# ------------------------------------------------------------------- #
# # telegram
# adaptive
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop

# seasons
# autumn-TM01  spring-SP01  summer-SM01  winter-WN01

# style
# bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01

# subject
# landscape-LN01  travel-TR01  travel2-TR11

# ------------------------------------------------------------------- #

# # twitter
# adaptive
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop

# seasons
# autumn_TM01  spring_SP01  summer_SM01  winter_WN01

# style
# bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01

# subject
# landscape-LN01  travel-TR01  travel2-TR11

# ------------------------------------------- #

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter_WN01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases test --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases both --dataset style/cinematic_CN01 --ft  --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic2_CN11 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_nodown --phases test --dataset style/filminspired_warmgold --ft --r50unfreezeL4 # unfreeze L4 when FT
 
# python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic_FT01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
 
# python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage_VN01 --ft --r50unfreezeL4 # unfreeze L4 when FT


# -------------------------------------------------- #
# --------------------- R50_TF --------------------- #
# -------------------------------------------------- #

# FT model -> settings.freeze
# FT with L4 unfroze -> unfreezeL4
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn_TM01 --ft

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn_TM01 --ft 

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/spring_SP01 --ft  

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/summer_SM01 --ft  

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/winter_WN01 --ft 

# python3 launcher.py --detector R50_TF --phases both --dataset style/bw_BW01 --ft  

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic_CN01 --ft  

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic2_CN11 --ft 

# python3 launcher.py --detector R50_TF --phases both --dataset style/filminspired_warmgold --ft 
 
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic_FT01 --ft  
 
# python3 launcher.py --detector R50_TF --phases both --dataset style/vintage_VN01 --ft 

# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg_strong --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/enhance_portrait --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/sky_bluedrama --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/subject_pop --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset style/film_inspired_boldbw --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset style/film_inspired_coolbw --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape_LN01 --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset subject/travel_TR01 --ft 
# python3 launcher.py --detector R50_TF --phases both --dataset subject/travel2_TR11 --ft 


# ------------------------------------------- #
# ---- Unfreezing layer 4 in R50 when FT ---- #
# ------------------------------------------- #


# ---------------------------- #
# -------- R50_nodonw -------- #
# ---------------------------- #

# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/blurbg_strong --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/enhance_portrait --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset adaptive/subject_pop --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset style/film_inspired_boldbw --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset style/film_inspired_coolbw --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset subject/landscape_LN01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset subject/travel_TR01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_nodown --phases both --dataset subject/travel2_TR11 --ft --r50unfreezeL4


#!/usr/bin/env bash

# ---------------------------- #
# ---------- R50_TF ---------- #
# ---------------------------- #

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/spring_SP01 --ft --r50unfreezeL4 

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4 

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/winter_WN01 --ft --r50unfreezeL4 

# python3 launcher.py --detector R50_TF --phases both --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic_CN01 --ft  --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic2_CN11 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset style/filminspired_warmgold --ft --r50unfreezeL4 # unfreeze L4 when FT
 
# python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic_FT01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
 
# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage_VN01 --ft --r50unfreezeL4 # unfreeze L4 when FT # training not complete

# # missing 

# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg_strong --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/enhance_portrait --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset adaptive/subject_pop --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_boldbw --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset style/film_inspired_coolbw --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape_LN01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset subject/travel_TR01 --ft --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset subject/travel2_TR11 --ft --r50unfreezeL4

# adaptive
# blurbg-subtle  blurbg_strong  enhance_portrait  sky_bluedrama  subject_pop

# subject
# landscape_LN01  travel2_TR11  travel_TR01

# ------------------------------------------------------------------- #
# facebook
# seasons
# autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/spring-SP01 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social facebook

# # style
# # bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# python3 launcher.py --detector R50_TF --phases both --dataset style/bw-BW01 --ft  --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic-CN01 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic2-CN11 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social facebook
# ------------------------------------------------------------------ #
# wrong susbset for FORLAB????d
# python3 launcher.py --detector R50_TF --phases both --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social facebook 
# check on telegram and twitter to see if also them have problems
# ------------------------------------------------------------------ #
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases both --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social facebook

# adaptive 
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social facebook


# subject
# landscape-LN01  travel-TR01  travel2-TR11
python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social facebook
python3 launcher.py --detector R50_TF --phases both --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social facebook

# ------------------------------------------------------------------- #
# facebook
# seasons
# autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# style
# bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# adaptive 
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# subject
# landscape-LN01  travel-TR01  travel2-TR11

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
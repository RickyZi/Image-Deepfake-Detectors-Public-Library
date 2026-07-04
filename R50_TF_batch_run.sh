#!/usr/bin/env bash

# ---------------------------- #
# ---------- R50_TF ---------- #
# ---------------------------- #

python3 launcher.py --detector R50_TF --phases both --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 # unfreeze L4 when FT # training not compelte

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/spring_SP01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/summer_SM01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset seasons/winter_WN01 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic_CN01 --ft  --r50unfreezeL4

# python3 launcher.py --detector R50_TF --phases both --dataset style/cinematic2_CN11 --ft --r50unfreezeL4 # unfreeze L4 when FT

# python3 launcher.py --detector R50_TF --phases both --dataset style/filminspired_warmgold --ft --r50unfreezeL4 # unfreeze L4 when FT
 
# python3 launcher.py --detector R50_TF --phases both --dataset style/futuristic_FT01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
 
python3 launcher.py --detector R50_TF --phases both --dataset style/vintage_VN01 --ft --r50unfreezeL4 # unfreeze L4 when FT # training not complete

# missing 

python3 launcher.py --detector R50_TF --phases both --dataset adaptive/blurbg_strong --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset adaptive/enhance_portrait --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset adaptive/subject_pop --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset style/film_inspired_boldbw --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset style/film_inspired_coolbw --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset subject/landscape_LN01 --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset subject/travel_TR01 --ft --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases both --dataset subject/travel2_TR11 --ft --r50unfreezeL4

# trained but wrong testing
python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 # unfreeze L4 when FT

python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4 # unfreeze L4 when FT

python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter_WN01 --ft --r50unfreezeL4 # unfreeze L4 when FT

python3 launcher.py --detector R50_TF --phases test --dataset style/bw_BW01 --ft  --r50unfreezeL4 # unfreeze L4 when FT

python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01 --ft  --r50unfreezeL4

python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2_CN11 --ft --r50unfreezeL4 # unfreeze L4 when FT

python3 launcher.py --detector R50_TF --phases test --dataset style/filminspired_warmgold --ft --r50unfreezeL4 # unfreeze L4 when FT
 
python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic_FT01 --ft  --r50unfreezeL4 # unfreeze L4 when FT
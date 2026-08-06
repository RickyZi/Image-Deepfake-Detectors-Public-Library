#!/usr/bin/env bash
# ------------------------------------------ #
# --------------- FT model ----------------- #
# ------------------------------------------ #
# seasons
# autumn_TM01/ spring_SP01/ summer_SM01/ winter_WN01/
# seasons
# autumn-TM01  spring-SP01  summer-SM01  winter-WN01
python3 launcher.py --detector NPR --phases both --dataset seasons/autumn_TM01  --ft 
python3 launcher.py --detector NPR --phases both --dataset seasons/spring_SP01  --ft 
python3 launcher.py --detector NPR --phases both --dataset seasons/summer_SM01  --ft 
python3 launcher.py --detector NPR --phases both --dataset seasons/winter_WN01  --ft 

# # style
# # bw_BW01/  cinematic2_CN11/  cinematic_CN01/ film_inspired_boldbw/   film_inspired_coolbw/   film_inspired_warmgold/ futuristic_FT01/  vintage_VN01/
python3 launcher.py --detector NPR --phases both --dataset style/bw_BW01  --ft  
python3 launcher.py --detector NPR --phases both --dataset style/cinematic_CN01  --ft 
python3 launcher.py --detector NPR --phases both --dataset style/cinematic2_CN11  --ft 
python3 launcher.py --detector NPR --phases both --dataset style/film_inspired_warmgold  --ft 
python3 launcher.py --detector NPR --phases both --dataset style/film_inspired_boldbw  --ft 
python3 launcher.py --detector NPR --phases both --dataset style/film_inspired_coolbw  --ft  # check this!!!
python3 launcher.py --detector NPR --phases both --dataset style/futuristic_FT01  --ft  
python3 launcher.py --detector NPR --phases both --dataset style/vintage_VN01  --ft 

# # adaptive 
# # blurbg_strong/    blurbg_subtle/    enhance_portrait/ sky_bluedrama/    subject_pop/
python3 launcher.py --detector NPR --phases both --dataset adaptive/blurbg_strong  --ft  
python3 launcher.py --detector NPR --phases both --dataset adaptive/blurbg_subtle  --ft 
python3 launcher.py --detector NPR --phases both --dataset adaptive/enhance_portait  --ft 
python3 launcher.py --detector NPR --phases both --dataset adaptive/sky_bluedrama  --ft 
python3 launcher.py --detector NPR --phases both --dataset adaptive/subject_pop  --ft 


# # subject
# # landscape_LN01/ travel2_TR11/   travel_TR01/
python3 launcher.py --detector NPR --phases both --dataset subject/landscape_LN01  --ft 
python3 launcher.py --detector NPR --phases both --dataset subject/travel_TR01  --ft  
python3 launcher.py --detector NPR --phases both --dataset subject/travel2_TR11  --ft 
# #!/usr/bin/env bash


# # R50_nodown -> trained only classification head
# # python3 launcher.py --detector R50_nodown --phases both --dataset seasons/autumn_TM01 --ft
# # ['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR']

# # runned on baseline tf2k dataset -> to be extendet to all datasets


# # -------------------------------------------------- #
# # --------------------- R50_TF --------------------- #
# # -------------------------------------------------- #

# # # python3 launcher.py --detector R50_TF --phases test

# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg_strong

# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance_portrait

# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky_bluedrama

# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject_pop

# # python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_boldbw

# # python3 launcher.py --detector R50_TF --phases test --dataset style/film_inspired_coolbw

# # python3 launcher.py --detector R50_TF --phases test --dataset subject/landscape_LN01

# # python3 launcher.py --detector R50_TF --phases test --dataset subject/travel_TR01

# # python3 launcher.py --detector R50_TF --phases test --dataset subject/travel2_TR11

# # # python3 launcher.py --detector R50_TF --phases test --dataset style/vintage_VN01

# # test TF_social
# python3 launcher.py --detector R50_TF --phases test --dataset Facebook 
# python3 launcher.py --detector R50_TF --phases test --dataset Telegram
# python3 launcher.py --detector R50_TF --phases test --dataset Twitter


# # # --------------------------------------------------- #
# # # ----------------------- NPR ----------------------- #
# # # --------------------------------------------------- #

# # # python3 launcher.py --detector NPR --phases test 

# # python3 launcher.py --detector NPR --phases test --dataset adaptive/blurbg_strong

# # python3 launcher.py --detector NPR --phases test --dataset adaptive/enhance_portrait

# # python3 launcher.py --detector NPR --phases test --dataset adaptive/sky_bluedrama

# # python3 launcher.py --detector NPR --phases test --dataset adaptive/subject_pop

# # python3 launcher.py --detector NPR --phases test --dataset style/film_inspired_boldbw

# # python3 launcher.py --detector NPR --phases test --dataset style/film_inspired_coolbw

# # python3 launcher.py --detector NPR --phases test --dataset subject/landscape_LN01

# # python3 launcher.py --detector NPR --phases test --dataset subject/travel_TR01

# # python3 launcher.py --detector NPR --phases test --dataset subject/travel2_TR11

# # test TF_social
# python3 launcher.py --detector NPR --phases test --dataset Facebook 
# python3 launcher.py --detector NPR --phases test --dataset Telegram
# python3 launcher.py --detector NPR --phases test --dataset Twitter
# # # --------------------------------------------------- #
# # # ----------------------- P2G ----------------------- #
# # # --------------------------------------------------- #

# # # python3 launcher.py --detector P2G --phases test  # still some trouble loading dataset

# # # python3 launcher.py --detector P2G --phases test --dataset adaptive/blurbg_strong

# # # python3 launcher.py --detector P2G --phases test --dataset adaptive/enhance_portrait

# # # python3 launcher.py --detector P2G --phases test --dataset adaptive/sky_bluedrama

# # # python3 launcher.py --detector P2G --phases test --dataset adaptive/subject_pop

# # # python3 launcher.py --detector P2G --phases test --dataset style/film_inspired_boldbw

# # # python3 launcher.py --detector P2G --phases test --dataset style/film_inspired_coolbw

# # # python3 launcher.py --detector P2G --phases test --dataset subject/landscape_LN01

# # # python3 launcher.py --detector P2G --phases test --dataset subject/travel_TR01

# # # python3 launcher.py --detector P2G --phases test --dataset subject/travel2_TR11

# # # test TF_social
# python3 launcher.py --detector P2G --phases test --dataset Facebook 
# python3 launcher.py --detector P2G --phases test --dataset Telegram
# python3 launcher.py --detector P2G --phases test --dataset Twitter

# # -------------------------------------------------- #
# # --------------------- R50_nd --------------------- #
# # -------------------------------------------------- #

# # # python3 launcher.py --detector R50_TF --phases test

# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg_strong

# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance_portrait

# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky_bluedrama

# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject_pop

# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film_inspired_boldbw

# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film_inspired_coolbw

# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape_LN01

# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel_TR01

# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2_TR11

# # # python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage_VN01

# # # test TF_social
# python3 launcher.py --detector R50_nodown --phases test --dataset Facebook 
# python3 launcher.py --detector R50_nodown --phases test --dataset Telegram
# python3 launcher.py --detector R50_nodown --phases test --dataset Twitter 

# # -------------------------------------------------- #
# # --------------------- CLIP-D --------------------- #
# # -------------------------------------------------- #

# # # python3 launcher.py --detector CLIP-D --phases test

# # python3 launcher.py --detector CLIP-D --phases test --dataset adaptive/blurbg_strong

# # python3 launcher.py --detector CLIP-D --phases test --dataset adaptive/enhance_portrait

# # python3 launcher.py --detector CLIP-D --phases test --dataset adaptive/sky_bluedrama

# # python3 launcher.py --detector CLIP-D --phases test --dataset adaptive/subject_pop

# # python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_boldbw

# # python3 launcher.py --detector CLIP-D --phases test --dataset style/film_inspired_coolbw

# # python3 launcher.py --detector CLIP-D --phases test --dataset subject/landscape_LN01

# # python3 launcher.py --detector CLIP-D --phases test --dataset subject/travel_TR01

# # python3 launcher.py --detector CLIP-D --phases test --dataset subject/travel2_TR11

# # # python3 launcher.py --detector CLIP-D --phases test --dataset style/vintage_VN01

# # # test TF_social
# python3 launcher.py --detector CLIP-D --phases test --dataset Facebook 
# python3 launcher.py --detector CLIP-D --phases test --dataset Telegram
# python3 launcher.py --detector CLIP-D --phases test --dataset Twitter


# ------------------------------------------ #
# R50 nodown #
# ------------------------------------------ #
# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn-TM01 --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/spring-SP01 --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer-SM01  --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter-WN01  --social facebook

# # style
# # bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# python3 launcher.py --detector R50_nodown --phases test --dataset style/bw-BW01  --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic-CN01 --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic2-CN11 --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-warmgold --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-boldbw --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-coolbw --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic-FT01  --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage-VN01 --social facebook

# # adaptive 
# # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-strong  --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-subtle --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance-portait --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky-bluedrama --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject-pop --social facebook

# baseline to test
python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-strong --social facebook
python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance-portait  --social facebook
python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky-bluedrama  --social facebook
python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject-pop  --social facebook


# # subject
# # landscape-LN01  travel-TR01  travel2-TR11
# python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape-LN01 --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel-TR01  --social facebook
# python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2-TR11 --social facebook


# ------------------------------------------ #
# R50 TF #
# ------------------------------------------ #

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn-TM01 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring-SP01 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer-SM01  --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter-WN01  --social facebook

# # style
# # bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# python3 launcher.py --detector R50_TF --phases test --dataset style/bw-BW01  --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic-CN01 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2-CN11 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-warmgold --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-boldbw --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-coolbw --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic-FT01  --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage-VN01 --social facebook

# adaptive 
# blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-strong  --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-subtle --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance-portait --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky-bluedrama --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject-pop --social facebook
python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-strong --social facebook
python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance-portait  --social facebook
python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky-bluedrama  --social facebook
python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject-pop  --social facebook


# # subject
# # landscape-LN01  travel-TR01  travel2-TR11
# python3 launcher.py --detector R50_TF --phases test --dataset subject/landscape-LN01 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel-TR01  --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel2-TR11 --social facebook
# #!/usr/bin/env bash

# subject
# landscape-LN01  travel-TR01  travel2-TR11
python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social twitter
python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social twitter
python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social twitter

python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social twitter
python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social twitter 
python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social twitter
python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social twitter

# # ---------------------------- #
# # ---------- R50_TF ---------- #
# # ---------------------------- #

# # facebook
# # seasons
# # # autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring-SP01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social facebook

# # # # # style
# # # # # bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# # python3 launcher.py --detector R50_TF --phases test --dataset style/bw-BW01 --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic-CN01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2-CN11 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social facebook # TRAINING THIS RN! (16:41)
# # python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social facebook

# # adaptive 
# # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social facebook # check!!
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social facebook


# # subject
# # landscape-LN01  travel-TR01  travel2-TR11
# python3 launcher.py --detector R50_TF --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social facebook
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social facebook

# # ------------------------------------------------------------------- #

# # ------------------------------------------------------------------- #
# # # telegram
# # seasons
# # # autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social telegram

# # # style
# # # bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# # python3 launcher.py --detector R50_TF --phases test --dataset style/bw01 --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/cinemantic2_CN11 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social telegram 
# # python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social telegram

# # # adaptive
# # # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social telegram
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social telegram

# # # subject
# # # landscape-LN01  travel-TR01  travel2-TR11
# # python3 launcher.py --detector R50_TF --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_TF --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social telegram


# # ------------------------------------------------------------------- #

# # # twitter
# # adaptive
# # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_TF --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social twitter

# # seasons
# # autumn_TM01  spring_SP01  summer_SM01  winter_WN01
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter_WN01 --ft --r50unfreezeL4 --social twitter

# # style
# # bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# python3 launcher.py --detector R50_TF --phases test --dataset style/bw01 --ft  --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/cinemantic2_CN11 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social twitter 
# python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social twitter

# # subject
# # landscape-LN01  travel-TR01  travel2-TR11
# python3 launcher.py --detector R50_TF --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social twitter
# python3 launcher.py --detector R50_TF --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social twitter

# # # ---------------------------- #
# # # ---------- R50_ND ---------- #
# # # ---------------------------- #

# # # facebook
# # # seasons
# # # # autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/spring-SP01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social facebook

# # # # style
# # # # bw-BW01  cinematic-CN01  cinematic2-CN11  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/bw-BW01 --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic-CN01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic2-CN11 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social facebook 
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social facebook

# # # # adaptive 
# # # # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social facebook

# # # # subject
# # # # landscape-LN01  travel-TR01  travel2-TR11
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social facebook
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social facebook




# # # ------------------------------------------------------------------- #
# # # # telegram
# # # seasons
# # # autumn-TM01  spring-SP01  summer-SM01  winter-WN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn-TM01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 --social telegram 
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer-SM01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter-WN01 --ft --r50unfreezeL4 --social telegram

# # # style
# # # bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/bw01 --ft  --r50unfreezeL4 --social telegramw
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic_CN01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinemantic2_CN11 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social telegram 
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social telegram

# # # adaptive
# # # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social telegram

# # # subject
# # # landscape-LN01  travel-TR01  travel2-TR11
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social telegram
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social telegram


# # # ------------------------------------------------------------------- #

# # # # twitter
# # # adaptive
# # # blurbg-strong  blurbg-subtle  enhance-portait  sky-bluedrama  subject-pop
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-strong --ft  --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/blurbg-subtle --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/enhance-portait --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/sky-bluedrama --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset adaptive/subject-pop --ft --r50unfreezeL4 --social twitter

# # # seasons
# # # autumn_TM01  spring_SP01  summer_SM01  winter_WN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/autumn_TM01 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/spring_SP01 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/summer_SM01 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset seasons/winter_WN01 --ft --r50unfreezeL4 --social twitter

# # # style
# # # bw01  cinemantic2_CN11  cinematic_CN01  film-inspired-boldbw  film-inspired-coolbw  film-inspired-warmgold  futuristic-FT01  vintage-VN01
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/bw01 --ft  --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinematic_CN01 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/cinemantic2_CN11 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-warmgold --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-boldbw --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/film-inspired-coolbw --ft --r50unfreezeL4 --social twitter 
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/futuristic-FT01 --ft  --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset style/vintage-VN01 --ft --r50unfreezeL4 --social twitter

# # # subject
# # # landscape-LN01  travel-TR01  travel2-TR11
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/landscape-LN01 --ft --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel-TR01 --ft  --r50unfreezeL4 --social twitter
# # python3 launcher.py --detector R50_nodown --phases test --dataset subject/travel2-TR11 --ft --r50unfreezeL4 --social twitter

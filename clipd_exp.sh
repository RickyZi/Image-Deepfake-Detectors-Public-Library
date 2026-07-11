#!/usr/bin/env bash
# ----------- #
# CLIP-D LoRA #
# ----------- #

# ------------ #
# - FACEBOOK - # 
# ------------ #   
# seasons
# autumn-TM01/ spring-SP01/ summer-SM01/ winter-WN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn-TM01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/spring-SP01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/summer-SM01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/winter-WN01 --social facebook

# # style
# # bw-BW01/                cinematic-CN01/         cinematic2-CN11/        film-inspired-boldbw/   film-inspired-coolbw/   film-inspired-warmgold/ futuristic-FT01/        vintage-VN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw-BW01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic-CN01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2-CN11 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-warmgold --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-coolbw --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-boldbw --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/futuristic-FT01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/vintage-VN01 --social facebook

# # # adaptive
# # # blurbg-strong/   blurbg-subtle/   enhance-portait/ sky-bluedrama/   subject-pop/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-strong --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-subtle --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/enhance-portait --social facebook
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky-bluedrama --social facebook
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject-pop --social facebook

# # # # subject
# # # # landscape-LN01/ travel-TR01/    travel2-TR11/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape-LN01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel-TR01 --social facebook
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2-TR11 --social facebook


# ------------ #
# - TELEGRAM - # 
# ------------ #   
# seasons
# # autumn-TM01/ spring-SP01/ summer-SM01/ winter-WN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn-TM01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/spring_SP01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/summer-SM01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/winter-WN01 --social telegram

# # style
# # # bw-BW01/                cinematic-CN01/         cinematic2-CN11/        film-inspired-boldbw/   film-inspired-coolbw/   film-inspired-warmgold/ futuristic-FT01/        vintage-VN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic_CN01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-warmgold --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-coolbw --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-boldbw --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/futuristic-FT01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/vintage-VN01 --social telegram

# # # adaptive
# # # blurbg-strong/   blurbg-subtle/   enhance-portait/ sky-bluedrama/   subject-pop/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-strong --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-subtle --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/enhance-portait --social telegram
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky-bluedrama --social telegram
python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject-pop --social telegram

# # # # subject
# # # # landscape-LN01/ travel-TR01/    travel2-TR11/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape-LN01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel-TR01 --social telegram
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2-TR11 --social telegram




# # ------------- #
# # -- TWITTER -- # 
# # ------------- #   
# # seasons
# # autumn_TM01/ spring_SP01/ summer_SM01/ winter_WN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/autumn_TM01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/spring_SP01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/summer_SM01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset seasons/winter_WN01 --social twitter

# # # style
# # # bw01/                   cinematic_CN01/         film-inspired-coolbw/   futuristic-FT01/
# # # cinemantic2_CN11/       film-inspired-boldbw/   film-inspired-warmgold/ vintage-VN01/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/bw01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic_CN01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/cinematic2_CN11 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-warmgold --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-coolbw --social twitter 
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/film-inspired-boldbw --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/futuristic-FT01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset style/vintage-VN01 --social twitter

# # adaptive
# # blurbg-strong/   blurbg-subtle/   enhance-portait/ sky-bluedrama/   subject-pop/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-strong --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/blurbg-subtle --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/enhance-portait --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/sky-bluedrama --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset adaptive/subject-pop --social twitter

# # # subject
# # # landscape-LN01/ travel-TR01/    travel2-TR11/
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/landscape-LN01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel-TR01 --social twitter
# python launcher.py --detector CLIP-D --phases test  --weights-name lora_r4_qv --tf2k True --ft --dataset subject/travel2-TR11 --social twitter
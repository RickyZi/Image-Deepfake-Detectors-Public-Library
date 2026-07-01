#!/usr/bin/env bash
# run multiple demo in batch, e.g. for different presets

# ---------------------------- #
# test baseline #
# ---------------------------- #

# # CLIP-D
# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector CLIP-D --phases test --dataset sedasons/spring_SP01

# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector CLIP-D --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/bw_BW01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector CLIP-D --phases test --dataset style/filminspired_warmgold

# python3 launcher.py --detector CLIP-D --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector CLIP-D --phases test --dataset style/vintage_VN01

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
 
python3 launcher.py --detector CLIP-D --phases both --dataset style/vintage_VN01 --ft --mlp

#!/usr/bin/env bash


# R50_nodown -> trained only classification head
# python3 launcher.py --detector R50_nodown --phases both --dataset seasons/autumn_TM01 --ft
# ['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR']

# runned on baseline tf2k dataset -> to be extendet to all datasets


# -------------------------------------------------- #
# --------------------- R50_TF --------------------- #
# -------------------------------------------------- #

# python3 launcher.py --detector R50_TF --phases test

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/spring_SP01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector R50_TF --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector R50_TF --phases test --dataset style/bw_BW01

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector R50_TF --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector R50_TF --phases test --dataset style/filminspired_warmgold

# python3 launcher.py --detector R50_TF --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector R50_TF --phases test --dataset style/vintage_VN01


# # --------------------------------------------------- #
# # ----------------------- NPR ----------------------- #
# # --------------------------------------------------- #

# # python3 launcher.py --detector NPR --phases test 

# python3 launcher.py --detector NPR --phases test --dataset seasons/autumn_TM01

# python3 launcher.py --detector NPR --phases test --dataset seasons/spring_SP01

# python3 launcher.py --detector NPR --phases test --dataset seasons/summer_SM01

# python3 launcher.py --detector NPR --phases test --dataset seasons/winter_WN01

# python3 launcher.py --detector NPR --phases test --dataset style/bw_BW01

# python3 launcher.py --detector NPR --phases test --dataset style/cinematic_CN01

# python3 launcher.py --detector NPR --phases test --dataset style/cinematic2_CN11

# python3 launcher.py --detector NPR --phases test --dataset style/filminspired_warmgold

# python3 launcher.py --detector NPR --phases test --dataset style/futuristic_FT01

# python3 launcher.py --detector NPR --phases test --dataset style/vintage_VN01

# --------------------------------------------------- #
# ----------------------- P2G ----------------------- #
# --------------------------------------------------- #

# python3 launcher.py --detector P2G --phases test  # still some trouble loading dataset

python3 launcher.py --detector P2G --phases test --dataset seasons/autumn_TM01

python3 launcher.py --detector P2G --phases test --dataset seasons/spring_SP01

python3 launcher.py --detector P2G --phases test --dataset seasons/summer_SM01

python3 launcher.py --detector P2G --phases test --dataset seasons/winter_WN01

python3 launcher.py --detector P2G --phases test --dataset style/bw_BW01

python3 launcher.py --detector P2G --phases test --dataset style/cinematic_CN01

python3 launcher.py --detector P2G --phases test --dataset style/cinematic2_CN11

python3 launcher.py --detector P2G --phases test --dataset style/filminspired_warmgold

python3 launcher.py --detector P2G --phases test --dataset style/futuristic_FT01

python3 launcher.py --detector P2G --phases test --dataset style/vintage_VN01

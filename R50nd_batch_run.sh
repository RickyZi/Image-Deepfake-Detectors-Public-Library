
python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/blurbg_subtle --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/enhance_portrait --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/sky_bluedrama --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset adaptive/subject_pop --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset style/film_inspired_boldbw --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset style/film_inspired_coolbw --ft --r50unfreezeL4
# python3 launcher.py --detector R50_TF --phases train --dataset subject/landscape_LN01 --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset subject/travel_TR01 --ft --r50unfreezeL4
python3 launcher.py --detector R50_nodown --phases train --dataset subject/travel2_TR11 --ft --r50unfreezeL4
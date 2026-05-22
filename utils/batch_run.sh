#!/usr/bin/env bash
# run multiple demo in batch, e.g. for different presets

#  season/SM01
# echo "Running demo for season/SM01 preset..."
python3 launcher.py --demo --demo-dataset season_SM01 --demo-detector P2G

#  season/SP01
echo "Running demo for season/SP01 preset..."
python3 launcher.py --demo --demo-dataset season_SP01 --demo-detector P2G

#  season/TM01
echo "Running demo for season/TM01 preset..."
python3 launcher.py --demo --demo-dataset season_TM01 --demo-detector P2G

#  season/WN01
echo "Running demo for season/WN01 preset..."
python3 launcher.py --demo --demo-dataset season_WN01 --demo-detector P2G

#  style/BW01
echo "Running demo for style/BW01 preset..."
python3 launcher.py --demo --demo-dataset style_BW01 --demo-detector P2G

#  style/CN01
echo "Running demo for style/CN01 preset..."
python3 launcher.py --demo --demo-dataset style_CN01 --demo-detector P2G

#  style/CN11
echo "Running demo for style/CN11 preset..."
python3 launcher.py --demo --demo-dataset style_CN11 --demo-detector P2G

#  style/FT01
echo "Running demo for style/FT01 preset..."
python3 launcher.py --demo --demo-dataset style_FT01 --demo-detector P2G

#  style/VN01
echo "Running demo for style/VN01 preset..."
python3 launcher.py --demo --demo-dataset style_VN01 --demo-detector P2G

#  style/warmgold
echo "Running demo for style/warmgold preset..."
python3 launcher.py --demo --demo-dataset style_warmgold --demo-detector P2G

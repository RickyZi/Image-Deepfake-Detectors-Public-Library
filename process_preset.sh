#!/usr/bin/env bash

set -euo pipefail

# This script processes images in a specified platform directory using the tf2k_org.py utility.
if [[ $# -ne 1 ]]; then # Check if exactly one argument is provided
    echo "Usage: $0 {pre|facebook|telegram|twitter}" >&2
    exit 2
fi

# Determine the platform based on the first argument
case "${1,,}" in
    pre)      platform="pre" ;;
    fb) platform="social/facebook" ;;
    tl) platform="social/telegram" ;;
    x)  platform="social/twitter" ;;
    *)
        echo "Error: unsupported platform '$1'." >&2
        echo "Choose one of: pre, facebook, telegram, twitter." >&2
        exit 2
        ;;
esac

# Get the directory of the script and change to it
# script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# cd "$script_dir"

# define the input root directory based on the platform
input_root="/mnt/c/Users/ricky/Documenti/Immagini/Lightroom Saved Photos/$platform"
if [[ ! -d "$input_root" ]]; then
    echo "Error: input directory not found: $input_root" >&2
    exit 1
fi

# run the python script
python3 utils/tf2k_org.py "$input_root"

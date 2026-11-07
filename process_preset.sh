#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {pre|facebook|telegram|twitter}" >&2
    exit 2
fi

case "${1,,}" in
    pre)      platform="pre" ;;
    facebook) platform="social/facebook" ;;
    telegram) platform="social/telegram" ;;
    twitter)  platform="social/twitter" ;;
    *)
        echo "Error: unsupported platform '$1'." >&2
        echo "Choose one of: pre, facebook, telegram, twitter." >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

input_root="/mnt/c/Users/ricky/Documenti/Immagini/Lightroom Saved Photos/$platform"
if [[ ! -d "$input_root" ]]; then
    echo "Error: input directory not found: $input_root" >&2
    exit 1
fi

python3 utils/tf2k_org.py "$input_root"

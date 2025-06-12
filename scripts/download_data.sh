#!/usr/bin/env bash

# download_data.sh - Download and organize training data
#
# Usage: ./scripts/download_data.sh [options] [data_dir]
#   --dnsmos       Download the DNSMOS dataset
#   --voicebank    Download the VoiceBank+DEMAND corpus
#   --librispeech  Download the Librispeech corpus
#   --daps         Download the DAPS corpus
#   --vctk         Download the VCTK corpus
#   -h, --help     Show this message
#   data_dir       Directory to store datasets (default: data)
#
# If no dataset flags are given, all datasets are downloaded.
#
# The script downloads several speech datasets and organizes them
# into foreground (clean speech) and background noise folders.

set -euo pipefail

# The script attempts to verify each URL before downloading using
# `wget --spider`.  However, some environments block outbound network
# access, which would cause the script to abort when the check fails.
# We therefore treat such failures as warnings and continue so that the
# user can retry or manually download the dataset later.

function usage() {
    cat <<EOF
Usage: $0 [options] [data_dir]
  --dnsmos       Download the DNSMOS dataset
  --voicebank    Download the VoiceBank+DEMAND corpus
  --librispeech  Download the Librispeech corpus
  --daps         Download the DAPS corpus
  --vctk         Download the VCTK corpus
  -h, --help     Show this help message
  data_dir       Directory to store datasets (default: data)

If no dataset flags are given, all datasets are downloaded.
EOF
}

DATA_DIR=data
declare -a DATASETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dnsmos|--voicebank|--librispeech|--daps|--vctk)
            DATASETS+=("${1#--}")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            DATA_DIR=$1
            shift
            ;;
    esac
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(dnsmos voicebank librispeech daps vctk)
fi

FG_DIR="$DATA_DIR/fg"
BG_DIR="$DATA_DIR/bg"
RIR_DIR="$DATA_DIR/rir"

mkdir -p "$FG_DIR/en" "$BG_DIR/en" "$RIR_DIR" "$DATA_DIR/raw"

function check_url() {
    local url=$1
    if ! wget --spider "$url" > /dev/null 2>&1; then
        echo "Warning: failed to access $url" >&2
        return 1
    fi
    return 0
}

function download_and_extract() {
    local url=$1
    local dest=$2
    mkdir -p "$dest"
    local fname=$(basename "$url")
    if [ ! -f "$dest/$fname" ]; then
        check_url "$url" || true
        if ! wget -c "$url" -O "$dest/$fname"; then
            echo "Failed to download $url" >&2
            rm -f "$dest/$fname"
            return 1
        fi
    fi
    case "$fname" in
        *.zip)
            unzip -n "$dest/$fname" -d "$dest" || return 1 ;;
        *.tar.gz|*.tgz)
            tar -xf "$dest/$fname" -C "$dest" || return 1 ;;
    esac
}

function copy_wavs() {
    local src=$1
    local dst=$2
    if [ ! -d "$src" ]; then
        echo "Warning: $src not found, skipping" >&2
        return
    fi
    mkdir -p "$dst"
    find "$src" -type f -name '*.wav' -exec cp -n {} "$dst" \;
}


function download_dnsmos() {
    DNSMOS_URL="https://github.com/microsoft/DNS-Challenge/raw/master/Datasets/DNSMOS/DNSMOS_dataset_48K.tar.gz"
    if [ ! -d "$DATA_DIR/raw/dnsmos/clean" ] || [ ! -d "$DATA_DIR/raw/dnsmos/noise" ]; then
        if ! download_and_extract "$DNSMOS_URL" "$DATA_DIR/raw/dnsmos"; then
            echo "Skipping DNSMOS dataset" >&2
            return
        fi
    fi
    copy_wavs "$DATA_DIR/raw/dnsmos/clean" "$FG_DIR/en"
    copy_wavs "$DATA_DIR/raw/dnsmos/noise" "$BG_DIR/en"
}

function download_voicebank() {
    VB_URL_CLEAN="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
    VB_URL_NOISE="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
    if [ ! -d "$DATA_DIR/raw/voicebank/clean_trainset_28spk_wav" ] || [ ! -d "$DATA_DIR/raw/voicebank/noisy_trainset_28spk_wav" ]; then
        if ! download_and_extract "$VB_URL_CLEAN" "$DATA_DIR/raw/voicebank" || ! download_and_extract "$VB_URL_NOISE" "$DATA_DIR/raw/voicebank"; then
            echo "Skipping VoiceBank dataset" >&2
            return
        fi
    fi
    copy_wavs "$DATA_DIR/raw/voicebank/clean_trainset_28spk_wav" "$FG_DIR/en"
    copy_wavs "$DATA_DIR/raw/voicebank/noisy_trainset_28spk_wav" "$BG_DIR/en"
}

function download_librispeech() {
    LIBRI_URL="https://openslr.elda.org/resources/12/train-clean-100.tar.gz"
    if [ ! -d "$DATA_DIR/raw/librispeech/LibriSpeech/train-clean-100" ]; then
        if ! download_and_extract "$LIBRI_URL" "$DATA_DIR/raw/librispeech"; then
            echo "Skipping Librispeech dataset" >&2
            return
        fi
    fi
    copy_wavs "$DATA_DIR/raw/librispeech/LibriSpeech/train-clean-100" "$FG_DIR/en"
}

function download_daps() {
    DAPS_URL="https://zenodo.org/record/3527842/files/daps.tar.gz?download=1"
    if [ -z "$(find "$DATA_DIR/raw/daps" -name '*.wav' -print -quit 2>/dev/null)" ]; then
        if ! download_and_extract "$DAPS_URL" "$DATA_DIR/raw/daps"; then
            echo "Skipping DAPS dataset" >&2
            return
        fi
    fi
    copy_wavs "$DATA_DIR/raw/daps" "$FG_DIR/en"
}

function download_vctk() {
    VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    if [ ! -d "$DATA_DIR/raw/vctk/wav48" ]; then
        if ! download_and_extract "$VCTK_URL" "$DATA_DIR/raw/vctk"; then
            echo "Skipping VCTK dataset" >&2
            return
        fi
    fi
    copy_wavs "$DATA_DIR/raw/vctk/wav48" "$FG_DIR/en"
}

for ds in "${DATASETS[@]}"; do
    case $ds in
        dnsmos) download_dnsmos ;;
        voicebank) download_voicebank ;;
        librispeech) download_librispeech ;;
        daps) download_daps ;;
        vctk) download_vctk ;;
    esac
done

cat <<'MSG'
Download script finished. If downloads succeeded, clean speech is under
\$FG_DIR and noise under \$BG_DIR. Check the messages above for any skipped
datasets.
MSG


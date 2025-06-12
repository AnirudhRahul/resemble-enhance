#!/usr/bin/env bash

# download_data.sh - Download and organize training data
#
# Usage: ./scripts/download_data.sh [options] [data_dir]
#   --dnsmos       Download the DNSMOS dataset
#   --voicebank    Download the VoiceBank+DEMAND corpus
#   --librispeech  Download the Librispeech corpus
#   --daps         Download the DAPS corpus
#   --vctk         Download the VCTK corpus
#   data_dir: directory to store datasets (default: data)
#
# If no dataset flags are given, all datasets are downloaded.
#
# The script downloads several speech datasets and organizes them
# into foreground (clean speech) and background noise folders.

set -euo pipefail

DATA_DIR=data
declare -a DATASETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dnsmos|--voicebank|--librispeech|--daps|--vctk)
            DATASETS+=("${1#--}")
            shift
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

function download_and_extract() {
    url=$1
    dest=$2
    mkdir -p "$dest"
    fname=$(basename "$url")
    if [ ! -f "$dest/$fname" ]; then
        wget -c "$url" -O "$dest/$fname"
    fi
    case "$fname" in
        *.zip)
            unzip -n "$dest/$fname" -d "$dest" ;;
        *.tar.gz|*.tgz)
            tar -xf "$dest/$fname" -C "$dest" ;;
    esac
}

function copy_wavs() {
    src=$1
    dst=$2
    mkdir -p "$dst"
    find "$src" -type f -name '*.wav' -exec cp -n {} "$dst" \;
}


function download_dnsmos() {
    DNSMOS_URL="https://example.com/dnsmos_dataset_48k.tar.gz" # placeholder URL
    if [ ! -d "$DATA_DIR/raw/dnsmos" ]; then
        download_and_extract "$DNSMOS_URL" "$DATA_DIR/raw/dnsmos"
    fi
    copy_wavs "$DATA_DIR/raw/dnsmos/clean" "$FG_DIR/en"
    copy_wavs "$DATA_DIR/raw/dnsmos/noise" "$BG_DIR/en"
}

function download_voicebank() {
    VB_URL_CLEAN="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
    VB_URL_NOISE="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
    if [ ! -d "$DATA_DIR/raw/voicebank" ]; then
        download_and_extract "$VB_URL_CLEAN" "$DATA_DIR/raw/voicebank"
        download_and_extract "$VB_URL_NOISE" "$DATA_DIR/raw/voicebank"
    fi
    copy_wavs "$DATA_DIR/raw/voicebank/clean_trainset_28spk_wav" "$FG_DIR/en"
    copy_wavs "$DATA_DIR/raw/voicebank/noisy_trainset_28spk_wav" "$BG_DIR/en"
}

function download_librispeech() {
    LIBRI_URL="https://openslr.elda.org/resources/12/train-clean-100.tar.gz"
    if [ ! -d "$DATA_DIR/raw/librispeech" ]; then
        download_and_extract "$LIBRI_URL" "$DATA_DIR/raw/librispeech"
    fi
    copy_wavs "$DATA_DIR/raw/librispeech/LibriSpeech/train-clean-100" "$FG_DIR/en"
}

function download_daps() {
    DAPS_URL="https://zenodo.org/record/3527842/files/daps.tar.gz?download=1"
    if [ ! -d "$DATA_DIR/raw/daps" ]; then
        download_and_extract "$DAPS_URL" "$DATA_DIR/raw/daps"
    fi
    copy_wavs "$DATA_DIR/raw/daps" "$FG_DIR/en"
}

function download_vctk() {
    VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    if [ ! -d "$DATA_DIR/raw/vctk" ]; then
        download_and_extract "$VCTK_URL" "$DATA_DIR/raw/vctk"
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
All datasets downloaded. Clean speech stored in \$FG_DIR and noise in \$BG_DIR.
You may set hp.fg_dir and hp.bg_dir to these locations before training.
MSG


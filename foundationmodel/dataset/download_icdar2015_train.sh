#!/bin/bash
# Download ICDAR 2015 Challenge 2 Training set
# Creates ~/.cache/icdar2015/Challenge2_train/ directory structure

set -e

CACHE_DIR="$HOME/.cache/icdar2015"
CHALLENGE_DIR="$CACHE_DIR/Challenge2_train"
ZIP_FILE=$(mktemp)

trap "rm -f $ZIP_FILE" EXIT

echo "Creating cache directory..."
mkdir -p "$CHALLENGE_DIR"

echo "Downloading ICDAR 2015 Challenge 2 Training Images & Ground Truth..."
curl --insecure \
  --header 'Host: rrc.cvc.uab.es' \
  --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) Gecko/20100101 Firefox/140.0' \
  --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
  --header 'Accept-Language: en-US,en;q=0.5' \
  --referer 'https://rrc.cvc.uab.es/?ch=2&com=downloads' \
  --header 'DNT: 1' \
  --header 'Sec-GPC: 1' \
  --cookie 'PHPSESSID=l2i2d0rcjqu03rk92bmldhidnk' \
  --header 'Upgrade-Insecure-Requests: 1' \
  --header 'Sec-Fetch-Dest: document' \
  --header 'Sec-Fetch-Mode: navigate' \
  --header 'Sec-Fetch-Site: same-origin' \
  --header 'Sec-Fetch-User: ?1' \
  'https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task3_Images_GT.zip' \
  --output "$ZIP_FILE"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Failed to download from RRC website"
    exit 1
fi

echo "✓ Download complete"
echo "Extracting to $CHALLENGE_DIR..."

cd "$CHALLENGE_DIR"
unzip -q "$ZIP_FILE"

echo "✓ Extraction complete"
echo ""
echo "Dataset structure:"
ls -la "$CHALLENGE_DIR" | head -10
echo ""
echo "Ground truth files:"
ls -la "$CHALLENGE_DIR/gt/" | head -5
echo ""
echo "✓ ICDAR 2015 Training set download complete"

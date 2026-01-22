#!/bin/bash
# Download ICDAR 2015 Challenge 2 Test set
# Creates ~/.cache/icdar2015/Challenge2_test/ directory structure

set -e

CACHE_DIR="$HOME/.cache/icdar2015"
CHALLENGE_DIR="$CACHE_DIR/Challenge2_test"
ZIP_FILE=$(mktemp)
GT_FILE=$(mktemp)

trap "rm -f $ZIP_FILE $GT_FILE" EXIT

echo "Creating cache directory..."
mkdir -p "$CHALLENGE_DIR"

echo "Downloading ICDAR 2015 Challenge 2 Test Images..."
curl --header 'Host: rrc.cvc.uab.es' \
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
  'https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_Images.zip' \
  --output "$ZIP_FILE"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Failed to download test images from RRC website"
    exit 1
fi

echo "✓ Test images download complete"

echo "Downloading ICDAR 2015 Challenge 2 Test Ground Truth..."
curl --location \
  --header 'Host: rrc.cvc.uab.es' \
  --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) Gecko/20100101 Firefox/140.0' \
  'https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlMl9UZXN0X1Rhc2szX0dULnR4dA==' \
  --output "$GT_FILE"

if [ ! -f "$GT_FILE" ]; then
    echo "Error: Failed to download test ground truth from RRC website"
    exit 1
fi

echo "✓ Test ground truth download complete"

echo "Extracting images to $CHALLENGE_DIR..."
cd "$CHALLENGE_DIR"
unzip -q "$ZIP_FILE"

echo "✓ Image extraction complete"

echo "Copying ground truth file..."
cp "$GT_FILE" "$CHALLENGE_DIR/Challenge2_Test_Task3_GT.txt"

echo ""
echo "Dataset structure:"
ls -la "$CHALLENGE_DIR" | head -10
echo ""
echo "Ground truth file:"
ls -la "$CHALLENGE_DIR/Challenge2_Test_Task3_GT.txt"
echo ""
echo "Sample ground truth entries:"
head -5 "$CHALLENGE_DIR/Challenge2_Test_Task3_GT.txt"
echo ""
echo "✓ ICDAR 2015 Test set download complete"

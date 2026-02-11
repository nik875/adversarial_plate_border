#!/bin/bash

DIR=$1
OUTPUT=${2:-average_diff.png}

echo "Computing average difference zones for $DIR..."
python ../image_diff.py "$DIR" -o "$OUTPUT"
echo "Done!"

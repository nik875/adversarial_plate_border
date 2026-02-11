#!/bin/bash

cd $1

echo "Computing average difference zones for layer 10..."
python ../../image_diff.py "layer10_Final_Output_Vocab_Softmax" -o ../diff_layer10.png

echo "Computing average difference zones for layer 0..."
python ../../image_diff.py "layer0_Conv_Layer_1_32ch" -o ../diff_layer0.png

echo "Done! Generated diff_layer10.png and diff_layer0.png"

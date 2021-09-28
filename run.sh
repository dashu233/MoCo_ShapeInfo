#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py data/imagenet \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos --lr 0.015 --batch-size 128 \
  --output output/exp1_shape_info_for_moco  --epoch 200 \
  | tee -a stdout.txt 

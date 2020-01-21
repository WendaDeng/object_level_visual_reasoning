#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

# Pythonpath
PYTHONPATH=.

# Settings
resume=/mnt/shared_40t/dwd/epic-kitchens/orn/epic_checkpoint
root=/mnt/shared_40t/dwd/epic-kitchens/orn/data/epic
# mask_dir=/mnt/shared_40t/dwd/epic-kitchens/orn/data/epic/masks
class_type=verb

# Train the object head only with f=MLP
epochs=10
heads=object
python main.py --resume $resume \
--root $root \
# --mask_dir $mask_dir \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train \
--arch orn_two_heads \
--depth 50 \
-t 4 \
-b 32 \
--cuda \
--dataset epic \
--heads $heads \
--class-type $class_type \
--epochs $epochs \
--pooling avg \
--print-freq 100 \
--pooling avg \
--mask-confidence 0.5 \
-j 4

# Train the two heads with f=RNN and pooling is avg for context head
# epochs=10
# heads=object+context
# python main.py --resume $resume \
# --root $root \
# --mask_dir $mask_dir \
# --blocks 2D_2D_2D_2.5D \
# --object-head 2D \
# --add-background \
# --train-set train \
# --arch orn_two_heads \
# --depth 50 \
# -t 4 \
# -b 8 \
# --cuda \
# --dataset epic \
# --heads $heads \
# --class-type $class_type \
# --epochs $epochs \
# --print-freq 100 \
# --pooling rnn \
# --mask-confidence 0.5 \
# -j 4
#
# # Finally validate on the test set
# epochs=10
# heads=object+context
# python main.py --resume $resume \
# --root $root \
# --mask_dir $mask_dir \
# --blocks 2D_2D_2D_2.5D \
# --object-head 2D \
# --add-background \
# --train-set train \
# --arch orn_two_heads \
# --depth 50 \
# -t 4 \
# -b 10 \
# --cuda \
# --dataset epic \
# --heads $heads \
# --class-type $class_type \
# --epochs $epochs \
# --print-freq 100 \
# -j 4 \
# --pooling rnn \
# --nb-crops 8 \
# --mask-confidence 0.5 \
# -e

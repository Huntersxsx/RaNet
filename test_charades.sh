
export CUDA_VISIBLE_DEVICES=0,1

# test
python moment_localization/test.py --cfg experiments/charades/vgg-dynamic+gat.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/vgg-dot+8conv.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/i3d-raw-dynamic+gat.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/i3d-finetune-dynamic+gat.yaml --verbose --split test


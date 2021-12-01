export CUDA_VISIBLE_DEVICES=0,1,2,3

# python moment_localization/train.py --cfg experiments/charades/vgg-dynamic+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-dot+8conv.yaml --verbose
python moment_localization/train.py --cfg experiments/charades/vgg-dot+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-dynamic+8conv.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-raw-dynamic+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-finetune-dynamic+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/c3d-dynamic+gat.yaml --verbose

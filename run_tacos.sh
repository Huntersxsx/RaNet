# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python moment_localization/train.py --cfg experiments/tacos/dynamic+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/tacos/dot+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/tacos/dot+8conv.yaml --verbose
# python moment_localization/train.py --cfg experiments/tacos/dynamic+8conv.yaml --verbose
python moment_localization/train.py --cfg experiments/tacos/onlydynamic+8conv.yaml --verbose

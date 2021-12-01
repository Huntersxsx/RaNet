export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# test
python moment_localization/test.py --cfg experiments/tacos/dynamic+gat.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/tacos/dot+8conv.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/tacos/dot+gat.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/tacos/dynamic+8conv.yaml --verbose --split test

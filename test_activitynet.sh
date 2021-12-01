
export CUDA_VISIBLE_DEVICES=0,1,2,3

# test
python moment_localization/test.py --cfg experiments/activitynet/dynamic+gat.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/activitynet/dot+4conv.yaml --verbose --split test




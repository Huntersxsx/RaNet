 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python moment_localization/train.py --cfg experiments/activitynet/dynamic+gat.yaml --verbose
# python moment_localization/train.py --cfg experiments/activitynet/dot+4conv.yaml --verbose



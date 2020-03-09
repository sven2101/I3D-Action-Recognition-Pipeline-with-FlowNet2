# Finetune on split1 of RGB data of ucf101
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 rgb FlowNet2-css-ft-sd 1
# Finetune on split2 of flow data of ucf101
CUDA_VISIBLE_DEVICES=0 python finetune.py ucf101 flow FlowNet2-css-ft-sd 1
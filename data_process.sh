###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/home/liangzg/Project/Datasets/nuplan/nuplan-v1.1_mini/data/cache/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/home/liangzg/Project/Datasets/nuplan/nuplan-maps-v1.0/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/home/liangzg/Project/Datasets/nuplan/train_set" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \


#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/liuzhuangzhuang/DCASE_2020_TASK_5DATA/'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/'

# Hyper-parameters
GPU_ID=0
MODEL_TYPE=Space_Duo_Cnn_9_2max
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='train' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='validate' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='evaluate' --workspace=$WORKSPACE

# Calculate scalar
#python utils/features.py calculate_scalar --data_type='train' --workspace=$WORKSPACE


############ Train and validate on development dataset ############

# Train & inference
for TAXONOMY_LEVEL in  'fine' 'coarse'
do
  # Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=#$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --batch_size=#$BATCH_SIZE --cuda

   # Inference and evaluate
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --#workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --#iteration=10000 --batch_size=$BATCH_SIZE --cuda 
done


python utils/plot_statistics.py --workspace=$WORKSPACE --taxonomy_level='fine' --model_type=$MODEL_TYPE 
python utils/plot_statistics.py --workspace=$WORKSPACE --taxonomy_level='coarse' --model_type=$MODEL_TYPE 


#Plot model statistics comparion
#python utils/plot_results.py --workspace=$WORKSPACE --taxonomy_level=fine


############ Train on full development dataset and inference on evaluation dataset ############

# Train on full development dataset
for TAXONOMY_LEVEL in  'fine' 'coarse'
do
# Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold='none' --batch_size=$BATCH_SIZE --cuda
  
#   Inference
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_evaluation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold='none' --iteration=10000 --batch_size=$BATCH_SIZE --cuda
done

############ END ############

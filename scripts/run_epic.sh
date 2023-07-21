CUDA_VISIBLE_DEVICES=1,2,3,4 python -u /mnt/cephfs/home/alvin/jialiang/Drive-act/mvitv2/SlowFast-main/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/jialiang/Drive-act/mvitv2/SlowFast-main/configs/epick/MVITv2_B_32x3_epick.yaml\
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 4 \
    NUM_GPUS 4 \
    TEST.ENABLE False \
    MIXUP.ENABLE False  \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/jialiang/mvit/part_IRprompt_b4_drop0.3_lr4e-5 \
    TEST.BATCH_SIZE 8 \


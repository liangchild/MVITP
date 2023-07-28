CUDA_VISIBLE_DEVICES=3,4,5,6 python -u /mnt/cephfs/home/alvin/jialiang/Drive-act/mvit/MVITP/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/jialiang/Drive-act/mvit/MVITP/configs/epick/MVITv2_B_32x3_epick.yaml\
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 4 \
    NUM_GPUS 4 \
    TEST.ENABLE False \
    MIXUP.ENABLE False  \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/jialiang/mvit/partIR_eventprompt_b4_drop0.3_lr4e-5 \
    TEST.BATCH_SIZE 8 \


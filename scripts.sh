# Path to the pretrained model
MODEL_PATH='/home/comp/zhenshun/pretrained/umt_pretrained/b16_ptk710_f8_res224.pth'

# Output directory
OUTPUT_DIR='./output/vit_base'  # Replace with your actual output directory

# Set CUDA devices and other necessary environment variables
# Run the torchrun command with the specified arguments
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12821 \
    --nnodes=1 \
    main.py \
    --model vit_base \
    --nb_classes "5" \
    --in_chans "1"\
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR}/logs \
    --output_dir ${OUTPUT_DIR}/models \
    --num_workers 4 \
    --batch_size 4 \
    --input_size 224 \
    --save_ckpt_freq 100 \
    --opt adamw \
    --lr 2.5e-4 \
    --layer_decay 0.75 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 300 \
    --drop_path 0.2 \
    --val_freq 50 \
    --enable_deepspeed \
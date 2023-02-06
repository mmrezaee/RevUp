#!/bin/bash
current_dir=$(pwd)
args=(
    # Observation Probability (\epsilon in the paper)
    --obsv_prob 1.0\
    # Noise Probability (\eta in the paper)
    --noise_prob 0.0\
    --wandb_project 'deleteme'\
    # Using Cuda
    --cuda \
    # Token Embedding Size
    --emb_size 200\
    # Encoder Hidden Size
    --enc_hid_size 100\
    # Decoder Hidden Size
    --dec_hid_size 100\
    # Encoder and Decoder Number of Layers
    --nlayers 2\
    # Learning Rate
    --lr 0.001\
    --validate_after 2500\
    # Clipping the gradient
    --clip 5.0\
    # Number of Epochs
    --epochs 100\
    # Batch Size
    --batch_size 150\
    # Number of Clauses (m in the paper)
    --num_clauses 5\
    # Number of Frames
    --num_Frames 603\
    --dataset 'NYT'\
    --train_data './demo_datasets/NYT/train_demo.json'\
    --valid_data './demo_datasets/NYT/valid_demo.json'\
    # Vocabulary for tokens
    --vocab './demo_datasets/NYT/all_args_150005.pkl'\
    # Vocabulary for frames
    --frame_vocab_address './demo_datasets/NYT/args_semafor_603.pkl'\
    )
echo hostname-$(hostname)
python ./main.py "${args[@]}" 


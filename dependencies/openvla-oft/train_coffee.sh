export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_API_KEY=    # TODO: use your own WANDB_API_KEY
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
 --vla_path openvla/openvla-7b \
 --data_root_dir ./data_files/sft_data/tensorflow_datasets \
 --dataset_name coffee_d0_300_demos \
 --run_root_dir ./checkpoint_files/openvla-oft/coffee_d0_300_demos \
 --use_l1_regression False \
 --use_diffusion False \
 --use_film False \
 --num_images_in_input 1 \
 --use_proprio False \
 --batch_size 8 \
 --learning_rate 5e-4 \
 --num_steps_before_decay 100000 \
 --max_steps 150000 \
 --save_freq 10000 \
 --save_latest_checkpoint_only False \
 --image_aug True \
 --lora_rank 32 \
 --wandb_entity "YOUR_WANDB_ENTITY" \
 --wandb_project "openvla-oft" \
 --run_id_note coffee_d0_300_demos

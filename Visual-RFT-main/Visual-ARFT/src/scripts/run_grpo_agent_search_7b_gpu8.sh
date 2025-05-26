export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./log_qwen25vl_7b_grpo_agent_search_data20_63_gpu8.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /src/visual_arft/src/open_r1/grpo_agent_search.py \
    --output_dir /share_models/Qwen2.5-VL-7B-Instruct_GRPO_agent_search_data20_63_gpu8 \
    --model_name_or_path /share_model/Qwen2.5-VL-7B-Instruct \
    --dataset_name /train_data/rft_agent_20.json \
    --deepspeed /src/visual_arft/local_scripts/zero3_offload.json \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 400 \
    --run_name Qwen25-VL-7B-GRPO-Agent-Search-data20-63-gpu8 \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8
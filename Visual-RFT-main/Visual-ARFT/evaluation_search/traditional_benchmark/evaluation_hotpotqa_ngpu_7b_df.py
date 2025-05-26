import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

model_path = "/shared/mllm_ckpts/models--Qwen--Qwen2.5-VL-7B-Instruct"

def run(rank, world_size):
    ### 多卡时候，device_map需要设置为cpu，再分配到不同GPU上，不能设置为 auto
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='cpu',
    )
    processor = AutoProcessor.from_pretrained(model_path)

    model = model.to(torch.device(rank))
    model = model.eval()

    # 加载数据集数据
    wikimultihopqa = []
    with open('/hotpotqa/dev.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                wikimultihopqa.append(json.loads(line))
    print(len(wikimultihopqa))

    # wikimultihopqa = wikimultihopqa[:4]

    print(wikimultihopqa[0]['question'])
    print(wikimultihopqa[0]['golden_answers'])
    print("Rank:" + str(rank))
    print("World Size:" + str(world_size))
    import math
    split_length = math.ceil(len(wikimultihopqa)/world_size)
    print("Split Chunk Length:" + str(split_length))
    split_wikimultihopqa = wikimultihopqa[int(rank*split_length) : int((rank+1)*split_length)]
    print(len(split_wikimultihopqa))
    wikimultihopqa = split_wikimultihopqa

    combine_results = []
    for i in tqdm(range(len(wikimultihopqa))):
        pred_answer = None
        query = wikimultihopqa[i]['question']
        answer = wikimultihopqa[i]['golden_answers']
        item_id = wikimultihopqa[i]['id']
        # + 'Answer the question directly.'
        input_text = query + '\n' + 'Only Return the answer.'
        # print("################################################################")
        # print(query)
        # print(answer)
        try:
            messages = [
                { "role": "user", 
                    "content": [{"type": "text", "text": input_text}]}
            ]
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                # images=image_inputs,
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0]
            # print(result)
            pred_answer = result

        except Exception as e:
            logger.info("ERROR OCCURES")
            logger.info({e})
        if pred_answer != None:
            combine_results.append(
                {'id': item_id, 'pred_answer': pred_answer, 'gt': answer, 'query': query}
            )
        else:
            combine_results.append(
                {'id': item_id, 'pred_answer': None, 'gt': answer, 'query': query}
            )
    with open(f"7b_ori_hotpotqa_direct_infere_{rank}.json", "w", encoding="utf-8") as f:
        json.dump(combine_results, f, ensure_ascii=False, indent=4)
    return combine_results

def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_results = []
        for i in range(world_size):
            global_results = global_results + result_lists[i]
            
        logger.info("Done")
        logger.info('finished running')
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
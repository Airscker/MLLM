import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


model_path = "/shared/mllm_ckpts/models--Qwen--Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='cuda:0',
)
processor = AutoProcessor.from_pretrained(model_path)

with open('/share_data/MAT/MAT-Benchmark/MAT-Coding.json', 'r') as file:
    wikimultihopqa = json.load(file)
print(len(wikimultihopqa))

combine_results = []
for i in tqdm(range(len(wikimultihopqa))):
    pred_answer = None
    query = wikimultihopqa[i]['question']
    answer = wikimultihopqa[i]['answer']
    image_path = wikimultihopqa[i]['image_path']
    input_image_path = '/MAT/MAT-Benchmark/MAT-Coding-image/' + image_path
    item_id = wikimultihopqa[i]['id']

    # ### Direct Inference
    input_text = query + '\n' + "Answer the question directly. The answer should be very brief."

    ### CoT
    # input_text = SYSTEM_PROMPT + '\n' + query + "\n" + "You must output your thinking processs in <think> </think>. The answer between <answer> </answer> should be very brief."

    print(RED+input_text+RESET)
    print(GREEN+str(answer)+RESET)
    
    try:
        ###### 进行一轮推理 ######
        messages = [
            { "role": "user", "content": [{"type": "image","image": input_image_path}, {"type": "text", "text": input_text}]}
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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
        
        pred_answer = result
        print(YELLOW+result+RESET)

        # match = re.search(r"<answer>(.*?)</answer>", result, re.DOTALL)
        # if match:
        #     pred_answer = match.group(1).strip()
        # print(YELLOW+result+RESET)
        # print(YELLOW+pred_answer+RESET)

    except Exception as e:
        print("ERROR OCCURES")
        print({e})
    combine_results.append(
        {'pred_answer': pred_answer, 'gt': answer, 'query': query}
    )

print(len(combine_results))
with open(f"./7B_df.json", "w", encoding="utf-8") as f:
    json.dump(combine_results, f, ensure_ascii=False, indent=4)
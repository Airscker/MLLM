import os
import re
import string
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL

from sentence_transformers import SentenceTransformer, util
sentence_tranformers_model = SentenceTransformer('/fs-computility/mllm/shared/liuziyu/share_models/all-MiniLM-L6-v2')  # light weight


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    # Pink adaptor arguments
    adapter_llm_enable: bool = field(default=False, metadata={"help": "Enable LLM adaptors"})
    adapter_llm_dim: int = field(default=8, metadata={"help": "Hidden dimension for LLM adaptors"})
    adapter_llm_scale: float = field(default=1.0, metadata={"help": "Scale factor for LLM adaptors"})
    adapter_llm_dropout: float = field(default=0.05, metadata={"help": "Dropout rate for LLM adaptors"})
    adapter_vision_enable: bool = field(default=False, metadata={"help": "Enable vision adaptors"})
    adapter_vision_dim: int = field(default=8, metadata={"help": "Hidden dimension for vision adaptors"})
    adapter_vision_scale: float = field(default=1.0, metadata={"help": "Scale factor for vision adaptors"})
    adapter_vision_dropout: float = field(default=0.05, metadata={"help": "Dropout rate for vision adaptors"})
    adapter_attn: bool = field(default=True, metadata={"help": "Apply adaptors to attention layers"})
    adapter_mlp: bool = field(default=False, metadata={"help": "Apply adaptors to MLP layers"})
    adapter_non_linear: bool = field(default=False, metadata={"help": "Use non-linear activation in adaptors"})
    # clip_select_layer is not directly used in Qwen2VL model loading with adaptors in the same way as PinkModel,
    # as Qwen2VL handles its vision encoder internally. We include it for completeness if needed later.
    clip_select_layer: int = field(default=-2, metadata={"help": "CLIP select layer (less relevant for Qwen2VL direct integration)"})
    # Refinement loop arguments
    enable_refinement_loop: bool = field(default=False, metadata={"help": "Enable refinement loop with critic."})
    refinement_threshold_tau: float = field(default=0.5, metadata={"help": "Score threshold for refinement."})
    max_refinement_loops: int = field(default=3, metadata={"help": "Maximum number of refinement loops."})

def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_similarity(prediction, ground_truth, sentence_tranformers_model=sentence_tranformers_model):
    emb1 = sentence_tranformers_model.encode(prediction, convert_to_tensor=True)
    emb2 = sentence_tranformers_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.cos_sim(emb1, emb2)
    return float(cosine_score)

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                if '<answer>' in sol:
                    if '<search>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<answer>(.*?)</answer>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()
                        
                        # Compare the extracted answers
                        reward = compute_f1(student_answer, ground_truth)
                    
                elif '<search>' in sol:
                    if '<answer>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<search>(.*?)</search>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<search>(.*?)</search>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()
                        
                        # Compare the extracted answers
                        reward = compute_similarity(student_answer, ground_truth)

                        if reward<0.5:
                            reward = 0.0
                    
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion matches exactly one valid format."""
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_search = r"<think>.*?</think>\s*<search>.*?</search>"

    completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content in completion_contents:
        if content.count("<answer>")>=2 or content.count("<search>")>=2 or content.count("<think>")>=2:
            rewards.append(0.0)
        elif '<answer>' in content and '</answer>' in content:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            match_search = re.fullmatch(pattern_search, content, re.DOTALL)
            if match_answer and not match_search:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        elif '<search>' in content and '</search>' in content:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            match_search = re.fullmatch(pattern_search, content, re.DOTALL)
            if match_search and not match_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

SYSTEM_PROMPT_AGENT = """# Role  
You are a step-by-step multimodal reasoning assistant.  
Given an image, a question, and optional partial reasoning chain, your task is to solve the problem **one substep at a time**.  

# Guiding Principles  
At each turn, you must **either**:  
1. Issue **one specific, text-only search** enclosed in <search> </search> tags,  
2. Or provide the **final answer** enclosed in <answer> </answer> tags.  

All outputs **must begin with a thought** enclosed in <think> </think> tags, explaining your current reasoning and what to do next.  

- Do not reference “the image” in your searches.  
- Do not repeat past queries.  
- Only output **one action per step**: either <search> or <answer>, never both.  
- When ready to conclude, summarize reasoning and give a final answer.

# Output Format (strict):  
Always start with <think>. Do not output the previous reasoning chain. Then, depending on the case, output one of the following:

## 1. If reasoning continues:  
<think> Your current reasoning and next plan </think>  
<search> One precise, retrievable textual query </search>

## 2. If ready to conclude:  
<think> Summarize all reasoning and derive the answer </think>  
<answer> Final answer, as briefly as possible </answer>

# Current reasoning chain:
"""



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset——method 1
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset——method 2
    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    # Load the dataset——method 3
    from datasets import Dataset
    dataset = Dataset.from_json(script_args.dataset_name)

    def make_conversation_image(example):
        image_path = example['image_path']
        prompt = example['problem']
        solution = example['solution']
        formatted_conversation = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_AGENT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT_AGENT + '\n' + prompt},
                ],
            }
        ]
        # resize image to avoid OOM
        image = PIL.Image.open(image_path).resize((720,720))
        
        return {"image": image, "prompt": formatted_conversation, 'solution': solution}


    # if "image" in dataset[script_args.dataset_train_split].features:
    #     print("has image in dataset")
    #     dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    #     # dataset = dataset.remove_columns(["original_question", "original_answer"])

    # else:
    #     print("no image in dataset")
    #     dataset = dataset.map(make_conversation)
    #     dataset = dataset.remove_columns("messages")

    dataset = dataset.map(make_conversation_image)
    dataset = dataset.remove_columns(["image_path", "problem"])

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        # train_dataset=dataset[script_args.dataset_train_split],
        ### lzy modified
        train_dataset=dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        # Pass Pink adaptor arguments
        adapter_llm_enable=script_args.adapter_llm_enable,
        adapter_llm_dim=script_args.adapter_llm_dim,
        adapter_llm_scale=script_args.adapter_llm_scale,
        adapter_llm_dropout=script_args.adapter_llm_dropout,
        adapter_vision_enable=script_args.adapter_vision_enable,
        adapter_vision_dim=script_args.adapter_vision_dim,
        adapter_vision_scale=script_args.adapter_vision_scale,
        adapter_vision_dropout=script_args.adapter_vision_dropout,
        adapter_attn=script_args.adapter_attn,
        adapter_mlp=script_args.adapter_mlp,
        adapter_non_linear=script_args.adapter_non_linear,
        # Pass refinement loop arguments
        enable_refinement_loop=script_args.enable_refinement_loop,
        refinement_threshold_tau=script_args.refinement_threshold_tau,
        max_refinement_loops=script_args.max_refinement_loops,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

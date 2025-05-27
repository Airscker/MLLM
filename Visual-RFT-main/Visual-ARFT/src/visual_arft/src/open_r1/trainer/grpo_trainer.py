# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

# Import Pink adaptors (now from the adapted local version)
from visual_arft.src.pink.pink_adapted_adapter import adapter, visual_adapter

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        # Pink adaptor arguments
        adapter_llm_enable: bool = False,
        adapter_llm_dim: int = 8,
        adapter_llm_scale: float = 1.0,
        adapter_llm_dropout: float = 0.05,
        adapter_vision_enable: bool = False,
        adapter_vision_dim: int = 8,
        adapter_vision_scale: float = 1.0,
        adapter_vision_dropout: float = 0.05,
        adapter_attn: bool = True,
        adapter_mlp: bool = False,
        adapter_non_linear: bool = False,
        # Refinement loop arguments
        enable_refinement_loop: bool = False,
        refinement_threshold_tau: float = 0.5,
        max_refinement_loops: int = 3,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                with adapter(hidden_dim=adapter_llm_dim, scale=adapter_llm_scale, dropout=adapter_llm_dropout, enabled=adapter_llm_enable, non_linear=adapter_non_linear, attn=adapter_attn, mlp=adapter_mlp):
                    with visual_adapter(hidden_dim=adapter_vision_dim, scale=adapter_vision_scale, dropout=adapter_vision_dropout, attn=adapter_attn, mlp=adapter_mlp, enabled=adapter_vision_enable, non_linear=adapter_non_linear):
                        model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                # Assuming Qwen2.5-VL might also benefit from similar adaptor structure
                with adapter(hidden_dim=adapter_llm_dim, scale=adapter_llm_scale, dropout=adapter_llm_dropout, enabled=adapter_llm_enable, non_linear=adapter_non_linear, attn=adapter_attn, mlp=adapter_mlp):
                    with visual_adapter(hidden_dim=adapter_vision_dim, scale=adapter_vision_scale, dropout=adapter_vision_dropout, attn=adapter_attn, mlp=adapter_mlp, enabled=adapter_vision_enable, non_linear=adapter_non_linear):
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                with adapter(hidden_dim=adapter_llm_dim, scale=adapter_llm_scale, dropout=adapter_llm_dropout, enabled=adapter_llm_enable, non_linear=adapter_non_linear, attn=adapter_attn, mlp=adapter_mlp):
                    with visual_adapter(hidden_dim=adapter_vision_dim, scale=adapter_vision_scale, dropout=adapter_vision_dropout, attn=adapter_attn, mlp=adapter_mlp, enabled=adapter_vision_enable, non_linear=adapter_non_linear):
                        self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                with adapter(hidden_dim=adapter_llm_dim, scale=adapter_llm_scale, dropout=adapter_llm_dropout, enabled=adapter_llm_enable, non_linear=adapter_non_linear, attn=adapter_attn, mlp=adapter_mlp):
                    with visual_adapter(hidden_dim=adapter_vision_dim, scale=adapter_vision_scale, dropout=adapter_vision_dropout, attn=adapter_attn, mlp=adapter_mlp, enabled=adapter_vision_enable, non_linear=adapter_non_linear):
                        self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # Store refinement loop parameters
        self.enable_refinement_loop = enable_refinement_loop
        self.refinement_threshold_tau = refinement_threshold_tau
        self.max_refinement_loops = max_refinement_loops

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        decoded_completions_ans0 = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Initialize final completions list, potentially to be refined
        final_completions_text = list(decoded_completions_ans0) # working with text for refinement simplicity

        if self.enable_refinement_loop:
            # Ensure unwrapped_model is available, as it's used inside the loop for refinement generation
            # This re-uses the `unwrapped_model` from the outer scope; nested `with` might not be needed if scope allows.
            # However, to be explicit and ensure it's the correct model instance:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model_for_refinement:
                # The original `prompts_text` and `images` are of size B (batch_size)
                # `decoded_completions_ans0` is of size B * G (num_generations)
                # We need to map each completion back to its original prompt and image
                for i in range(len(decoded_completions_ans0)):
                    original_prompt_idx = i // self.num_generations
                    current_question_text = prompts_text[original_prompt_idx] # Original question text
                    current_image_input = images[original_prompt_idx] # Original image
                    
                    ans_text = decoded_completions_ans0[i]
                    
                    for loop_idx in range(self.max_refinement_loops):
                        critique_result = placeholder_critic_evaluate(current_question_text, current_image_input, ans_text)
                        score = critique_result["score"]
                        critique = critique_result["critique"]

                        if score < self.refinement_threshold_tau:
                            # Construct prompt_prime for refinement
                            # This needs to be formatted according to model's expected conversational style
                            # We use a simplified template here. `maybe_apply_chat_template` might be needed
                            # depending on `self.processing_class` and model requirements.
                            # The SYSTEM_PROMPT_AGENT from grpo_agent_search.py is a reference for full structure.
                            
                            # Attempting a structure similar to make_conversation_image
                            # The 'SYSTEM_PROMPT_AGENT' or a similar base prompt might be needed here.
                            # For now, directly using the REFINEMENT_PROMPT_TEMPLATE text.
                            # This part is critical and might need adjustment based on how Qwen2VL processes chat.
                            
                            # Simplified refinement prompt text construction
                            refinement_prompt_text_input = REFINEMENT_PROMPT_TEMPLATE.format(
                                question_text=current_question_text,
                                initial_answer=ans_text,
                                critique=critique
                            )
                            
                            # Prepare input for the model - this is a simplified representation.
                            # The actual input preparation might need to follow the complex structure
                            # in `grpo_agent_search.py`'s `make_conversation_image` or `apply_chat_template`.
                            # Assuming `current_image_input` is a PIL Image.
                            # The prompt for refinement should be structured as a new user turn.
                            
                            # Let's try to mimic the conversational format.
                            # The `prompts` variable (original conversational prompts) has the structure.
                            # We need to find the original user prompt text to prepend.
                            # `inputs[original_prompt_idx]["prompt"]` is the original conversational structure.
                            
                            # For Qwen2VL, text input is usually part of a list of content dicts.
                            # Example: [{"type": "text", "text": "..."}]
                            # We need to ensure the refinement prompt text is correctly embedded.

                            # Using a simplified text-only refinement prompt for now.
                            # A more robust solution would re-construct the conversation history
                            # or use a dedicated refinement template that matches the model's chat format.
                            
                            # This processing step is crucial and complex:
                            # It needs to exactly match how the model expects inputs for multi-turn conversation with images.
                            # The `self.processing_class` handles image and text together.
                            # We are creating a new textual turn.
                            
                            # Option 1: Simple text prompt (might not leverage conversation history well)
                            # refinement_input_text = refinement_prompt_text_input

                            # Option 2: Try to format it like a conversational turn.
                            # This requires knowing the structure `self.processing_class` expects.
                            # `maybe_apply_chat_template` is usually for a full conversation.
                            # Here we are adding a new turn.
                            
                            # Let's assume the model can take a direct text string as a follow-up,
                            # given the image is already processed with the initial prompt.
                            # This is a strong assumption.

                            # A more robust way: Get the original conversation structure
                            # from `inputs[original_prompt_idx]['prompt']` (which is a list of dicts)
                            # and append a new user turn with the refinement_prompt_text_input.
                            # However, `inputs` is a list of dicts, not directly subscriptable like this.
                            # `prompts` contains the conversational structures.
                            
                            # The `prompts_text` was already processed by `maybe_apply_chat_template`.
                            # We are creating a new turn.
                            # For now, we make a simplified input for refinement.
                            # This is a key area for potential improvement.
                            
                            current_prompt_structure = prompts[original_prompt_idx] # This is List[Dict]
                            
                            # Create a new turn for refinement
                            # This is a guess on the structure. The actual structure might differ.
                            # It should be a list of messages if using chat template.
                            # For Qwen-VL, it's often:
                            # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "initial_q"}]}]
                            # For refinement, it would be:
                            # [{"role": "user", "content": [{"type": "text", "text": "refinement_text"}]}]
                            # The image is implicitly part of the "session" or needs to be re-passed.
                            
                            # We will re-pass the image with the new text.
                            # The refinement text itself should guide the model.

                            # For Qwen2VL, the processor expects a list of texts and a list of images.
                            # `current_image_input` is already a PIL.Image.
                            # `refinement_prompt_text_input` is a string.
                            refinement_processed_inputs = self.processing_class(
                                text=[refinement_prompt_text_input], # Must be a list
                                images=[current_image_input],    # Must be a list
                                return_tensors="pt",
                                padding=True, # Apply padding as per model's usual handling
                                padding_side="left", # Consistent with initial prompt padding
                                add_special_tokens=False # Assuming False based on initial generation
                            ).to(self.accelerator.device)


                            # Ensure generation config does not return multiple sequences for refinement
                            refinement_gen_config = copy.deepcopy(self.generation_config)
                            refinement_gen_config.num_return_sequences = 1 
                            # Ensure max_new_tokens is appropriate for generating a refined answer
                            refinement_gen_config.max_new_tokens = self.max_completion_length 

                            refined_completion_ids_full = unwrapped_model_for_refinement.generate(
                                input_ids=refinement_processed_inputs["input_ids"],
                                attention_mask=refinement_processed_inputs["attention_mask"],
                                pixel_values=refinement_processed_inputs["pixel_values"], # Pass pixel_values
                                image_grid_thw=refinement_processed_inputs["image_grid_thw"], # Pass image_grid_thw
                                generation_config=refinement_gen_config,
                            )
                            
                            # The refined_completion_ids_full includes the input prompt part. We need to slice it.
                            refinement_input_len = refinement_processed_inputs["input_ids"].shape[1]
                            refined_completion_ids_only = refined_completion_ids_full[:, refinement_input_len:]

                            refined_ans_text_list = self.processing_class.batch_decode(refined_completion_ids_only, skip_special_tokens=True)
                            
                            if refined_ans_text_list: # Check if list is not empty
                                ans_text = refined_ans_text_list[0] # Take the first (and only) refined answer
                                final_completions_text[i] = ans_text # Update the list of completions
                            else:
                                # Handle case of empty generation, though unlikely with proper setup
                                break # Break refinement loop if generation fails

                        else: # Score is good enough
                            break
                    # End of refinement loop for one completion
            # End of loop over all completions
            
        # `final_completions_text` now contains original or refined answers.
        # Convert to conversational format if needed for reward computation
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": text}] for text in final_completions_text]
        else:
            completions = final_completions_text # Should be a list of strings

        # Compute the rewards using the potentially refined completions
        # The `prompts` list is already prepared for B*G elements.
        # Ensure `completions` also matches this structure.
        # If `is_conversational` is true, `completions` is List[List[Dict]].
        # If not, `completions` is List[str].
        
        # `final_completions_text` now contains original or refined answers (list of B*G strings)
        # Convert to conversational format if needed for reward computation
        if is_conversational(inputs[0]): # `inputs` is the original batch data (list of B items)
            # `completions_for_reward` should be List[List[Dict]] of size B*G
            completions_for_reward = [[{"role": "assistant", "content": text}] for text in final_completions_text]
        else:
            # `completions_for_reward` should be List[str] of size B*G
            completions_for_reward = final_completions_text

        # Compute the rewards using the potentially refined completions
        # `prompts` (from the top of compute_loss) is List[Dict] of size B (original prompts with structure)
        # We need to expand it to match the B*G completions.
        # `inputs` is a list of dicts from the dataloader, one for each of B items.
        # `inputs[original_prompt_idx]['prompt']` gives the conversational structure for one original prompt.
        expanded_prompts_for_reward = [inputs[i // self.num_generations]['prompt'] 
                                      for i in range(len(completions_for_reward))]

        # Initialize rewards_per_func tensor correctly for B*G items
        rewards_per_func = torch.zeros(len(completions_for_reward), len(self.reward_funcs), device=device)
        
        for i_reward, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                reward_model_inputs_text = []
                # `expanded_prompts_for_reward` contains the structured prompt for each of the B*G items
                # `completions_for_reward` contains the corresponding completion for each of the B*G items
                if is_conversational(inputs[0]): # Check based on the original dataset format
                    for original_conv_prompt_structure, assistant_completion_structure in zip(expanded_prompts_for_reward, completions_for_reward):
                        # original_conv_prompt_structure is like [{"role": "user", ...}]
                        # assistant_completion_structure is like [{"role": "assistant", ...}]
                        full_conversation_structure = original_conv_prompt_structure + assistant_completion_structure
                        # self.processing_class.tokenizer.apply_chat_template or self.processing_class.apply_chat_template
                        # The exact method depends on the processor type. AutoProcessor might have apply_chat_template directly.
                        # Using self.processing_class.tokenizer.apply_chat_template as it's common for tokenizers.
                        if hasattr(self.processing_class, 'apply_chat_template'):
                            applied_template = self.processing_class.apply_chat_template(full_conversation_structure, tokenize=False, add_generation_prompt=False)
                        elif hasattr(self.processing_class.tokenizer, 'apply_chat_template'):
                             applied_template = self.processing_class.tokenizer.apply_chat_template(full_conversation_structure, tokenize=False, add_generation_prompt=False)
                        else:
                            raise ValueError("Processor or its tokenizer does not have apply_chat_template method.")
                        reward_model_inputs_text.append(applied_template)
                else: 
                    # If not conversational, `expanded_prompts_for_reward` should be list of text strings
                    # and `completions_for_reward` also list of text strings.
                    # This path requires careful handling of `expanded_prompts_for_reward` structure.
                    # `prompts_text` (B items) vs `expanded_prompts_for_reward` (B*G items from structured prompts).
                    # For simplicity, if not conversational, we assume `prompts_text` is the source for prompt part.
                    current_prompts_text_for_reward = [prompts_text[i // self.num_generations] for i in range(len(completions_for_reward))]
                    reward_model_inputs_text = [p_text + c_text for p_text, c_text in zip(current_prompts_text_for_reward, completions_for_reward)]
                
                reward_inputs = reward_processing_class(
                    reward_model_inputs_text, return_tensors="pt", padding=True, padding_side="right", 
                    truncation=True, max_length=reward_processing_class.model_max_length, # Add truncation
                    add_special_tokens=False 
                ).to(self.accelerator.device)
                
                with torch.inference_mode():
                    rewards_per_func[:, i_reward] = reward_func(**reward_inputs).logits.squeeze(-1) # ensure (B*G,)
            else: # Custom reward function
                # Rebuild reward_kwargs for B*G items from the original B items in `inputs`
                reward_kwargs_expanded = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion", "image"]}
                for key_r in reward_kwargs_expanded:
                    for example_original_batch_idx in range(len(inputs)): # Iterate B times (original batch)
                        reward_kwargs_expanded[key_r].extend([inputs[example_original_batch_idx][key_r]] * self.num_generations)
                
                output_reward_func = reward_func(prompts=expanded_prompts_for_reward, completions=completions_for_reward, **reward_kwargs_expanded)
                rewards_per_func[:, i_reward] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1) # Shape (B*G)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


def placeholder_critic_evaluate(question_text: str, image_input: Any, answer_ans0: str) -> dict:
    """
    Placeholder critic function.
    Inputs:
        question_text: str (the original textual question)
        image_input: Any (the image data, not used in this placeholder)
        answer_ans0: str (the initial answer from the generator)
    Outputs: A dictionary {"score": float, "critique": str}.
    """
    # Convert answer to lowercase for easier checking
    answer_lower = answer_ans0.lower()
    
    # Keywords that might indicate a low-quality answer
    low_quality_keywords = ["i don't know", "sorry", "cannot answer", "unable to determine", "not sure"]
    
    if not answer_ans0.strip(): # Check if the answer is empty or whitespace
        return {"score": 0.1, "critique": "The answer is empty."}
        
    for keyword in low_quality_keywords:
        if keyword in answer_lower:
            return {"score": 0.2, "critique": "The answer is not confident or incomplete."}
            
    # Example of a simple length check (very basic)
    if len(answer_ans0.split()) < 3: # If answer is less than 3 words
        return {"score": 0.4, "critique": "The answer is very short, potentially lacking detail."}

    # If no issues found by the placeholder logic
    return {"score": 0.9, "critique": "Looks good."}


# This will be used to format the prompt for refinement.
# It needs to align with how the model expects conversational input.
# We can adapt the SYSTEM_PROMPT_AGENT or a similar structure from grpo_agent_search.py
REFINEMENT_PROMPT_TEMPLATE = """Original Question: {question_text}
Initial Answer: {initial_answer}
Critique: {critique}
Please provide a refined answer based on the critique. Ensure your output starts with <think> and concludes with <answer> tags."""

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

import unittest
from unittest.mock import patch, MagicMock, call
import torch # Required for tensor operations in trainer if not fully mocked
import PIL.Image # For creating dummy image inputs

# Imports from the application code
# Adjust paths as necessary if tests are run from a different working directory
# Assuming PYTHONPATH or similar is set up for these imports to work
from visual_arft.src.open_r1.grpo_agent_search import GRPOScriptArguments
from visual_arft.src.open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer, placeholder_critic_evaluate, REFINEMENT_PROMPT_TEMPLATE
from trl import GRPOConfig, ModelConfig # For instantiating trainer args

# Dummy PIL Image for testing
def create_dummy_image():
    return PIL.Image.new('RGB', (60, 30), color = 'red')

class TestIntegrationFeatures(unittest.TestCase):

    def test_grpo_script_arguments_defaults(self):
        """Task 4: Test GRPOScriptArguments default values."""
        args = GRPOScriptArguments()
        self.assertFalse(args.adapter_llm_enable)
        self.assertEqual(args.adapter_llm_dim, 8)
        self.assertFalse(args.adapter_vision_enable)
        self.assertEqual(args.adapter_vision_dim, 8)
        self.assertTrue(args.adapter_attn)
        self.assertFalse(args.adapter_mlp)
        
        self.assertFalse(args.enable_refinement_loop)
        self.assertEqual(args.refinement_threshold_tau, 0.5)
        self.assertEqual(args.max_refinement_loops, 3)

    def test_placeholder_critic_evaluate(self):
        """Task 2: Test placeholder_critic_evaluate function."""
        dummy_image = create_dummy_image()
        
        # Test empty answer
        result = placeholder_critic_evaluate("question", dummy_image, "")
        self.assertEqual(result["score"], 0.1)
        self.assertEqual(result["critique"], "The answer is empty.")

        # Test answer with "sorry"
        result = placeholder_critic_evaluate("question", dummy_image, "I am sorry, I cannot answer.")
        self.assertEqual(result["score"], 0.2)
        self.assertEqual(result["critique"], "The answer is not confident or incomplete.")

        # Test very short answer
        result = placeholder_critic_evaluate("question", dummy_image, "Yes.")
        self.assertEqual(result["score"], 0.4)
        self.assertEqual(result["critique"], "The answer is very short, potentially lacking detail.")

        # Test a "good" answer
        result = placeholder_critic_evaluate("question", dummy_image, "This is a reasonable answer.")
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["critique"], "Looks good.")

    @patch('visual_arft.src.pink.pink_adapted_adapter.visual_adapter')
    @patch('visual_arft.src.pink.pink_adapted_adapter.adapter')
    @patch('transformers.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained') # Also mock processor loading
    def test_adaptor_loading_disabled(self, mock_auto_processor, mock_from_pretrained, mock_llm_adapter, mock_vision_adapter):
        """Task 1, Case 1: Test adaptors disabled."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.pad_token_id = 0
        mock_processor_instance.tokenizer.eos_token_id = 1
        mock_processor_instance.image_processor = MagicMock()

        mock_from_pretrained.return_value = mock_model_instance
        mock_auto_processor.return_value = mock_processor_instance
        
        # LLM adapter should be a context manager
        mock_llm_adapter_cm = MagicMock()
        mock_llm_adapter.return_value = mock_llm_adapter_cm
        
        # Vision adapter should be a context manager
        mock_vision_adapter_cm = MagicMock()
        mock_vision_adapter.return_value = mock_vision_adapter_cm

        training_args = GRPOConfig(output_dir="dummy_output")
        
        trainer = Qwen2VLGRPOTrainer(
            model="Qwen/Qwen2-VL-Chat", # dummy path
            reward_funcs=[], 
            args=training_args,
            adapter_llm_enable=False,
            adapter_vision_enable=False
        )
        
        # Check from_pretrained was called (it's always called)
        mock_from_pretrained.assert_called()

        # Assert context managers were entered with enabled=False or not at all if logic short-circuits
        # Based on current adapter.py, they are called but 'enabled' flag inside context manager controls behavior.
        mock_llm_adapter.assert_called_with(hidden_dim=unittest.mock.ANY, scale=unittest.mock.ANY, dropout=unittest.mock.ANY, enabled=False, non_linear=unittest.mock.ANY, attn=unittest.mock.ANY, mlp=unittest.mock.ANY)
        mock_vision_adapter.assert_called_with(hidden_dim=unittest.mock.ANY, scale=unittest.mock.ANY, dropout=unittest.mock.ANY, attn=unittest.mock.ANY, mlp=unittest.mock.ANY, enabled=False, non_linear=unittest.mock.ANY)


    @patch('visual_arft.src.pink.pink_adapted_adapter.visual_adapter')
    @patch('visual_arft.src.pink.pink_adapted_adapter.adapter')
    @patch('transformers.Qwen2VLForConditionalGeneration.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_adaptor_loading_enabled(self, mock_auto_processor, mock_from_pretrained, mock_llm_adapter, mock_vision_adapter):
        """Task 1, Case 2: Test adaptors enabled."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.pad_token_id = 0
        mock_processor_instance.tokenizer.eos_token_id = 1
        mock_processor_instance.image_processor = MagicMock()

        mock_from_pretrained.return_value = mock_model_instance
        mock_auto_processor.return_value = mock_processor_instance

        mock_llm_adapter_cm = MagicMock()
        mock_llm_adapter.return_value = mock_llm_adapter_cm
        
        mock_vision_adapter_cm = MagicMock()
        mock_vision_adapter.return_value = mock_vision_adapter_cm

        training_args = GRPOConfig(output_dir="dummy_output")
        
        trainer = Qwen2VLGRPOTrainer(
            model="Qwen/Qwen2-VL-Chat",
            reward_funcs=[],
            args=training_args,
            adapter_llm_enable=True,
            adapter_llm_dim=16,
            adapter_llm_scale=0.5,
            adapter_llm_dropout=0.1,
            adapter_vision_enable=True,
            adapter_vision_dim=32,
            adapter_vision_scale=0.8,
            adapter_vision_dropout=0.03,
            adapter_attn=False, # Test non-default
            adapter_mlp=True,   # Test non-default
            adapter_non_linear=True # Test non-default
        )
        
        mock_from_pretrained.assert_called()
        mock_llm_adapter.assert_called_with(hidden_dim=16, scale=0.5, dropout=0.1, enabled=True, non_linear=True, attn=False, mlp=True)
        mock_vision_adapter.assert_called_with(hidden_dim=32, scale=0.8, dropout=0.03, attn=False, mlp=True, enabled=True, non_linear=True)


    def _setup_refinement_loop_test(self, enable_loop, max_loops, tau):
        """Helper to set up a minimal trainer for refinement loop tests."""
        mock_model_instance = MagicMock()
        mock_model_instance.warnings_issued = {} # prevent error in trainer init
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.pad_token_id = 0
        mock_processor_instance.tokenizer.eos_token_id = 1
        mock_processor_instance.eos_token_id = 1 # for Qwen2VLGRPOTrainer internal use
        mock_processor_instance.image_processor = MagicMock()
        mock_processor_instance.batch_decode = MagicMock(side_effect=lambda x, **kwargs: [f"decoded_{i}" for i in range(x.size(0))] if hasattr(x, 'size') else ["decoded_0"])
        
        # Mock processing_class call for refinement
        mock_processor_instance_call_result = {
            "input_ids": torch.tensor([[1,2,3]]), 
            "attention_mask": torch.tensor([[1,1,1]]),
            "pixel_values": torch.randn(1,3,224,224), # Dummy pixel values
            "image_grid_thw": torch.tensor([[[1,1,1]]]) # Dummy image_grid_thw
        }
        mock_processor_instance.return_value = mock_processor_instance_call_result
        
        # Mock apply_chat_template if it's part of the processor or its tokenizer
        if hasattr(mock_processor_instance, 'apply_chat_template'):
            mock_processor_instance.apply_chat_template = MagicMock(return_value="formatted_prompt_for_reward")
        if hasattr(mock_processor_instance.tokenizer, 'apply_chat_template'):
            mock_processor_instance.tokenizer.apply_chat_template = MagicMock(return_value="formatted_prompt_for_reward")


        training_args = GRPOConfig(output_dir="dummy_output", per_device_train_batch_size=1) # batch size 1 for simplicity

        trainer = Qwen2VLGRPOTrainer(
            model=mock_model_instance, # Pass the mocked model instance
            reward_funcs=[], # No rewards needed for this control flow test
            args=training_args,
            processing_class=mock_processor_instance,
            enable_refinement_loop=enable_loop,
            max_refinement_loops=max_loops,
            refinement_threshold_tau=tau
        )
        trainer.accelerator = MagicMock() # Mock accelerator
        trainer.accelerator.device = torch.device('cpu')
        
        # Mock inputs for compute_loss
        # `inputs` is a list of dicts, each dict represents a sample from the dataset
        dummy_image = create_dummy_image()
        # `prompts` (passed to compute_loss) are derived from `inputs[x]['prompt']`
        # `prompts_text` are derived from `maybe_apply_chat_template(inputs[x]['prompt'], ...)`
        # `images` are derived from `inputs[x]['image']`
        
        # Let's make `inputs` simpler for this test, assuming it's already been processed to some extent
        # or that `maybe_apply_chat_template` and image processing work as expected.
        # The key is that `prompts_text` and `images` are available.
        
        # Mocking the structure that Qwen2VLGRPOTrainer.compute_loss expects for 'inputs'
        # Each item in 'inputs' list is a dictionary.
        # 'prompt' key holds the conversational structure.
        # 'image' key holds a PIL image.
        # 'solution' key (or others) might be present for reward functions but not essential for refinement loop logic itself.
        
        # For simplicity, we'll mock what `compute_loss` uses: `prompts_text` and `images`.
        # The `inputs` arg to `compute_loss` is a list of dicts.
        # `prompts = [x["prompt"] for x in inputs]`
        # `prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]`
        # `images = [x["image"] for x in inputs]`

        # To avoid mocking maybe_apply_chat_template, let's assume it returns the text directly
        # and `inputs[0]['prompt']` is what it would take.
        
        # We need to provide `prompts_text` and `images` to the mocked `compute_loss` context
        # `prompts_text` is a list of strings (batch_size)
        # `images` is a list of PIL Images (batch_size)
        # `inputs` for compute_loss should be a list of dicts, where each dict has 'prompt' and 'image'.
        # The 'prompt' in inputs should be the structured conversational prompt.
        
        # Simplified inputs for compute_loss
        # `prompts` (list of conversational dicts) and `images` (list of PIL Images) are extracted from `inputs`
        # `prompts_text` (list of strings) is also extracted and processed.

        # Let's prepare the direct inputs that the refinement loop will use
        # from within the mocked compute_loss context.
        # We'll patch `self.processing_class` directly on the trainer instance for more control.
        trainer.processing_class = mock_processor_instance


        return trainer, mock_model_instance # Return model to mock its generate method

    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.placeholder_critic_evaluate')
    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.unwrap_model_for_generation')
    def test_refinement_loop_max_times(self, mock_unwrap_model, mock_critic):
        """Task 3, Case 1: Loop runs max times."""
        trainer, model_mock = self._setup_refinement_loop_test(enable_loop=True, max_loops=3, tau=0.8)
        
        # Mock the behavior of the unwrapped model's generate method
        # Needs to handle initial generation and refinement generations
        mock_unwrapped_model_instance = MagicMock()
        # Initial generation (B*G sequences)
        initial_generation_output = torch.tensor([[1,2,3,4,5]]) # Example token IDs, B*G = 1*1 for simplicity
        # Refinement generations (1 sequence each time)
        refined_generation_output1 = torch.tensor([[10,11,12,13,14]])
        refined_generation_output2 = torch.tensor([[20,21,22,23,24]])
        refined_generation_output3 = torch.tensor([[30,31,32,33,34]])
        
        mock_unwrapped_model_instance.generate = MagicMock(side_effect=[
            initial_generation_output, 
            refined_generation_output1, 
            refined_generation_output2, 
            refined_generation_output3
        ])
        mock_unwrap_model.return_value.__enter__.return_value = mock_unwrapped_model_instance

        mock_critic.return_value = {"score": 0.5, "critique": "Needs more work."} # Always low score

        # Simplified inputs for compute_loss
        # `inputs` is a list of dicts.
        # `prompts_text` is a list of strings.
        # `images` is a list of PIL images.
        # `prompt_inputs` are the tokenized versions of these.
        
        # We need to construct the `inputs` argument for `compute_loss`
        # And also ensure that `prompt_inputs` are correctly formed for the first generation call
        
        # Let compute_loss run, it will use the mocked generate and critic
        # We need to provide the initial `inputs` that `compute_loss` expects.
        # These are constructed by the dataloader in a real scenario.
        # For this test, we provide minimal `inputs`.
        
        # `prompts`: list of conversational dicts
        # `prompts_text`: list of strings (processed from `prompts`)
        # `images`: list of PIL Images
        
        # Mocking the data `compute_loss` would receive and process
        # This setup is for a batch size of 1, num_generations = 1 for simplicity of tracking calls
        trainer.num_generations = 1 
        dummy_conv_prompt = [{"role": "user", "content": [{"type": "text", "text": "Original Question"}]}]
        test_inputs = [{'prompt': dummy_conv_prompt, 'image': create_dummy_image()}]

        # Mocking what self.processing_class(text=prompts_text, images=images, ...) would return
        # This is for the *initial* generation inside compute_loss
        trainer.processing_class.return_value = {
            "input_ids": torch.tensor([[1,2]]), # Dummy tokenized initial prompt
            "attention_mask": torch.tensor([[1,1]]),
            "pixel_values": torch.randn(1,3,224,224),
            "image_grid_thw": torch.tensor([[[1,1,1]]])
        }
        
        trainer.compute_loss(model=model_mock, inputs=test_inputs)

        self.assertEqual(mock_unwrapped_model_instance.generate.call_count, 1 + 3) # 1 initial + 3 refinement
        self.assertEqual(mock_critic.call_count, 3)


    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.placeholder_critic_evaluate')
    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.unwrap_model_for_generation')
    def test_refinement_loop_exits_early(self, mock_unwrap_model, mock_critic):
        """Task 3, Case 2: Loop exits early on good score."""
        trainer, model_mock = self._setup_refinement_loop_test(enable_loop=True, max_loops=3, tau=0.8)

        mock_unwrapped_model_instance = MagicMock()
        initial_generation_output = torch.tensor([[1,2,3,4,5]])
        refined_generation_output1 = torch.tensor([[10,11,12,13,14]])
        mock_unwrapped_model_instance.generate = MagicMock(side_effect=[initial_generation_output, refined_generation_output1])
        mock_unwrap_model.return_value.__enter__.return_value = mock_unwrapped_model_instance
        
        mock_critic.side_effect = [
            {"score": 0.5, "critique": "Needs more work."}, # First call: low score
            {"score": 0.9, "critique": "Looks good!"}      # Second call: high score
        ]

        trainer.num_generations = 1
        dummy_conv_prompt = [{"role": "user", "content": [{"type": "text", "text": "Original Question"}]}]
        test_inputs = [{'prompt': dummy_conv_prompt, 'image': create_dummy_image()}]
        trainer.processing_class.return_value = { # For initial generation
            "input_ids": torch.tensor([[1,2]]), "attention_mask": torch.tensor([[1,1]]),
            "pixel_values": torch.randn(1,3,224,224), "image_grid_thw": torch.tensor([[[1,1,1]]])
        }

        trainer.compute_loss(model=model_mock, inputs=test_inputs)

        self.assertEqual(mock_unwrapped_model_instance.generate.call_count, 1 + 1) # 1 initial + 1 refinement
        self.assertEqual(mock_critic.call_count, 2)


    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.placeholder_critic_evaluate')
    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.unwrap_model_for_generation')
    def test_refinement_loop_disabled(self, mock_unwrap_model, mock_critic):
        """Task 3, Case 3: Loop disabled."""
        trainer, model_mock = self._setup_refinement_loop_test(enable_loop=False, max_loops=3, tau=0.8)

        mock_unwrapped_model_instance = MagicMock()
        initial_generation_output = torch.tensor([[1,2,3,4,5]])
        mock_unwrapped_model_instance.generate = MagicMock(return_value=initial_generation_output)
        mock_unwrap_model.return_value.__enter__.return_value = mock_unwrapped_model_instance

        trainer.num_generations = 1
        dummy_conv_prompt = [{"role": "user", "content": [{"type": "text", "text": "Original Question"}]}]
        test_inputs = [{'prompt': dummy_conv_prompt, 'image': create_dummy_image()}]
        trainer.processing_class.return_value = { # For initial generation
            "input_ids": torch.tensor([[1,2]]), "attention_mask": torch.tensor([[1,1]]),
            "pixel_values": torch.randn(1,3,224,224), "image_grid_thw": torch.tensor([[[1,1,1]]])
        }
        
        trainer.compute_loss(model=model_mock, inputs=test_inputs)

        self.assertEqual(mock_unwrapped_model_instance.generate.call_count, 1) # Only initial generation
        mock_critic.assert_not_called()

    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.placeholder_critic_evaluate')
    @patch('visual_arft.src.open_r1.trainer.grpo_trainer.unwrap_model_for_generation')
    def test_refinement_prompt_construction(self, mock_unwrap_model, mock_critic):
        """Task 3, Case 4: Test prompt construction for refinement."""
        trainer, model_mock = self._setup_refinement_loop_test(enable_loop=True, max_loops=1, tau=0.8)
        
        mock_unwrapped_model_instance = MagicMock()
        initial_generation_output = torch.tensor([[1,2,3,4,5]]) # Corresponds to "decoded_0"
        refined_generation_output = torch.tensor([[10,11,12,13,14]])
        mock_unwrapped_model_instance.generate = MagicMock(side_effect=[initial_generation_output, refined_generation_output])
        mock_unwrap_model.return_value.__enter__.return_value = mock_unwrapped_model_instance

        mock_critic.return_value = {"score": 0.5, "critique": "Test Critique"}
        
        trainer.num_generations = 1
        original_question_text = "This is the original question?"
        # We need to ensure prompts_text in compute_loss gets this value.
        # This means the mocked call to maybe_apply_chat_template should produce it.
        # For simplicity, we'll ensure `prompts_text` within compute_loss has this.
        # This is tricky because `prompts_text` is derived inside `compute_loss`.
        # We can patch `maybe_apply_chat_template` or ensure `inputs` leads to this.
        
        # The `prompts_text` is derived from `inputs` using `maybe_apply_chat_template`.
        # Let's mock `maybe_apply_chat_template` for this specific test.
        with patch('visual_arft.src.open_r1.trainer.grpo_trainer.maybe_apply_chat_template', 
                   return_value={"prompt": original_question_text}) as mock_apply_template:

            dummy_conv_prompt = [{"role": "user", "content": [{"type": "text", "text": original_question_text}]}]
            test_inputs = [{'prompt': dummy_conv_prompt, 'image': create_dummy_image()}]
            
            # Mock for initial prompt processing
            trainer.processing_class.side_effect = [
                { # Initial prompt_inputs
                    "input_ids": torch.tensor([[1,2]]), "attention_mask": torch.tensor([[1,1]]),
                    "pixel_values": torch.randn(1,3,224,224), "image_grid_thw": torch.tensor([[[1,1,1]]])
                },
                { # Refinement prompt_inputs
                    "input_ids": torch.tensor([[10,11]]), "attention_mask": torch.tensor([[1,1]]),
                    "pixel_values": torch.randn(1,3,224,224), "image_grid_thw": torch.tensor([[[1,1,1]]]) # new dummy pixel values
                }
            ]
            # Mock for initial batch_decode
            trainer.processing_class.batch_decode = MagicMock(side_effect=lambda ids, **kwargs: ["initial_answer_0"] if ids.equal(torch.tensor([[3,4,5]])) else ["refined_answer_0"])


            trainer.compute_loss(model=model_mock, inputs=test_inputs)

            mock_apply_template.assert_called() # Ensure our prompt text setup was hit.

            # Check the arguments to the processing_class for the refinement call
            # The first call to processing_class is for the initial prompts.
            # The second call is for the refinement prompt.
            self.assertTrue(trainer.processing_class.call_count >= 2)
            refinement_call_args = trainer.processing_class.call_args_list[1][1] # kwargs of second call
            
            expected_refinement_text = REFINEMENT_PROMPT_TEMPLATE.format(
                question_text=original_question_text,
                initial_answer="initial_answer_0", # from the first batch_decode mock
                critique="Test Critique"
            )
            self.assertEqual(refinement_call_args['text'][0], expected_refinement_text)

if __name__ == '__main__':
    unittest.main()

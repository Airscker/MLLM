# Evaluation on MAT-Search
In this folder, we provide scripts for evaluation on **MAT-Search** and other four multihopQA Benchmark (2wikimultihopQA, HotpotQA, MuSiQue, Bamboogle).
``MAT-Search-Benchmark`` folder is used to evaluate the results on **MAT-Search**, while `traditional_benchmark` folder is used to evaluate the inference results of the other four multihopQA Benchmark.

## Download Dataset and Models
You can download MAT Benchmakr on 🤗<a href="https://huggingface.co/datasets/laolao77/MAT">Datasets</a>. Use `/MAT/MAT-Benchmark/MAT-Search.json`.

You can download our model: 🤗<a href="https://huggingface.co/laolao77/Visual-ARFT-Search">Visual-ARFT-Search</a></h3>

You can download other four multihopQA benchmark from 🤗<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets">Dataset</a></h3>.

## Web Search API
We use SerperAPI for web search. You can start by registering an account to receive 2,500 free queries, and then add your API key to the `.env` file.

## Inference on MAT-Search

Firstly, `cd MAT-Search-Benchmark`. In this folder, there are `evaluation_mat_search_ngpu_7b_df.py` and `evaluation_mat_search_ngpu_7b_visual_arft.py`. The first Python file is used to evaluate the results of direct inference, while the second Python file is used to evaluate the results of the **Visual-ARFT-Search** model.

To run `evaluation_mat_coding_visual_arft.py`, you need to replace the paths to the model and dataset:

-  Line 59: Replace **model_name** with the actual model path (**Visual-ARFT-Search**).
-  Line 76: Replace **json_path** with the actual dataset path (MAT-Search.json).
-  Line 97: Replace **input_image_path** with the actual image path.
-  Line 192: Set the results save path.

To run `evaluation_mat_search_ngpu_7b_df.py`, you need to replace the paths to the model and dataset:

-  Line 17: Replace **model_name** with the actual model path (**Original Qwen2.5-VL without Visual-ARFT**).
-  Line 27: Replace **json_path** with the actual dataset path (MAT-Search.json).
-  Line 37: Replace **input_image_path** with the actual image path.
-  Line 95: Set the results save path.

> ⏳ The code `evaluation_mat_coding_visual_arft.py` supports multi-GPU execution. The inference time for **Visual-ARFT-Coding-3/7B** is around ten minutes with 8GPUs.

## Inference on other MultihopQA Benchmark

Firstly, `cd traditional_benchmark`. Then, replace the paths as instructed above.

**2Wiki** contains over 12k questions and **HotpotQA** has over 7k, so the inference process may take a relatively long time. However, all scripts in the `traditional_benchmark` folder support multi-GPU inference.

## Evaluation
After obtaining the inference results, run the `evaluation.ipynb` in each folder step by step. The `.ipynb` file will provide the final evaluation scores.

We have also saved the inference results of the `Visual-ARFT-Search-7B` model in the `evaluation_results` folder. You can directly use the results inside to test the evaluation scores.








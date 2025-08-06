import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import json
from preprocess.prepare_data_safe import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from prompts import sys_prompt, cot_prompt, icl_sys_prompt, icl_cot_prompt, noevi_sys_prompt, noevi_cot_prompt, sys_prompt_gpt, cot_prompt_gpt
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original

def load_llama_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=None, dtype=torch.float16):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    # 修正 rope_scaling 欄位
    if hasattr(model.config, 'rope_scaling') and model.config.rope_scaling is not None:
        rope_scaling = model.config.rope_scaling
        if isinstance(rope_scaling, dict):
            model.config.rope_scaling = {
                "type": rope_scaling.get("type", "linear"),
                "factor": rope_scaling.get("factor", 1.0)
            }
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        return ckpt
    return []

def eval_all(pred_file_path, subset, split=None, eval_hops=-1):
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")
    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    print({f"results{postfix}/hit@1": hit1,
           f"results{postfix}/macro_f1": f1,
           f"results{postfix}/macro_precision": prec,
           f"results{postfix}/macro_recall": recall,
           f"results{postfix}/exact_match": em,
           f"results{postfix}/totally_wrong": tw,
           f"results{postfix}/micro_f1": mi_f1,
           f"results{postfix}/micro_precision": mi_prec,
           f"results{postfix}/micro_recall": mi_recall,
           f"results{postfix}/total_cnt": total_cnt,
           f"results{postfix}/no_ans_cnt": no_ans_cnt,
           f"results{postfix}/no_ans_ratio": no_ans_ratio,
           f"results{postfix}/hal_score": hal_score})
    if stats is not None:
        for k, v in stats.items():
            print({f"stats{postfix}/{k}": v})
    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    print({f"results{postfix}/hit": hit})
    print("=" * 50)

def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt

def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA (transformers Llama3-8B)")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    thres = args.thres

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{temperature}-thres_{thres}-{split}-predictions-resume.jsonl"

    if not os.path.exists(raw_pred_file_path):
        pred_file_path_for_data = ""
    else:
        pred_file_path_for_data = str(raw_pred_file_path)

    model, tokenizer = load_llama_model(model_name)
    data = get_data(dataset_name, pred_file_path_for_data, score_dict_path, split, prompt_mode)
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))
    with open(raw_pred_file_path, "a", encoding="utf-8") as pred_file:
        for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
            prompt = each_qa["user_query"]
            result = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            # 清理掉不必要欄位
            for k in ["graph", "good_paths_rog", "good_triplets_rog", "scored_triplets"]:
                if k in each_qa:
                    del each_qa[k]
            each_qa["prediction"] = result
            save_checkpoint(pred_file, each_qa)

    # If the processing completes, rename the files to remove the "resume" flag
    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    os.rename(raw_pred_file_path, final_pred_file_path)
    eval_all(final_pred_file_path, subset=True)
    eval_all(final_pred_file_path, subset=False)

if __name__ == "__main__":
    main()

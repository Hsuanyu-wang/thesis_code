# 匯入所需的標準與第三方套件
import os
import json
import wandb
import random
import argparse
import time
from tqdm import tqdm
from pathlib import Path
import glob
import datetime

# 匯入自定義模組
from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original

# 根據不同的 prompt_mode、model_name、llm_mode 選擇對應的 prompt
# 回傳系統提示詞與 COT（Chain-of-Thought）提示詞
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

# 將資料以 JSON 格式寫入檔案（用於儲存 checkpoint）
def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")

# 載入 checkpoint，回傳已處理的資料清單
def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []

# 評估所有預測結果，並將指標記錄到 wandb，且追加到 CSV
def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1, experiment_name=None, csv_path=None, reverse_order=False):
    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")

    print("Results:")
    # 使用修正版的評估指標
    # Note: pass split=None to evaluator so predictions are parsed by 'ans:' lines, not dataset split name
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=None, eval_hops=eval_hops, reverse_order=reverse_order)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    # 將各種評估指標記錄到 wandb（若有提供 run）
    if run is not None:
        run.log({f"results{postfix}/hit@1": hit1,
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
                 f"results{postfix}/hal_score": hal_score})  # score_h in the paper
        if stats is not None:
            for k, v in stats.items():
                run.log({f"stats{postfix}/{k}": v})

    # 使用原始版的 hit 指標
    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    if run is not None:
        run.log({f"results{postfix}/hit": hit})

    # 追加到 CSV（若指定 csv_path）
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "experiment_name", "subset", "split", "hit1", "hit", "macro_f1", "macro_precision", "macro_recall",
            "exact_match", "totally_wrong", "micro_f1", "micro_precision", "micro_recall", "total_cnt",
            "no_ans_cnt", "no_ans_ratio", "hal_score"
        ]
        # 格式與 aggregated_eval.csv 一致，對於未知的欄位使用空字串
        row = [
            experiment_name if experiment_name is not None else "",
            "sub" if subset else "full",
            split if split is not None else "",
            hit1, hit, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score
        ]
        write_header = not csv_path.exists()

        # 若已存在舊版 CSV（缺少 hit 欄位），自動升級：在 hit1 後插入 hit，舊資料以空字串補位
        if not write_header:
            try:
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    old_header = lines[0].strip().split(",")
                    if "hit" not in old_header:
                        if "hit1" in old_header:
                            insert_pos = old_header.index("hit1") + 1
                        else:
                            insert_pos = 3  # 預期位置（experiment_name, subset, split 之後）
                        new_header = old_header[:insert_pos] + ["hit"] + old_header[insert_pos:]
                        upgraded_lines = [",".join(new_header) + "\n"]
                        for line in lines[1:]:
                            cols = line.rstrip("\n").split(",")
                            # 將缺少欄位的行補齊
                            if len(cols) < len(new_header) - 1:
                                # 對極端不一致情況，直接跳過升級，保留原行
                                upgraded_lines.append(line)
                                continue
                            cols = cols[:insert_pos] + [""] + cols[insert_pos:]
                            upgraded_lines.append(",".join(cols) + "\n")
                        with open(csv_path, "w") as f:
                            f.writelines(upgraded_lines)
            except Exception:
                pass

        with open(csv_path, "a") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            # 確保將數值轉為字串
            f.write(",".join([str(x) for x in row]) + "\n")

    print("=" * 50)
    print("=" * 50)

def extract_experiment_id_from_path(path, dataset_name):
    """從路徑中提取實驗 ID。
    優先使用路徑的父資料夾名稱（例如：/home/.../webqsp_freq_weight_Sep02-01:05:30_patient_10/retrieval_result.pth -> webqsp_freq_weight_Sep02-01:05:30_patient_10），
    若不可用則回退到既有的正則（{dataset_name}_YYYYMMDD_HHMMSS）。"""
    if path is None:
        return None
    try:
        parent_dir_name = Path(str(path)).parent.name
        if parent_dir_name:
            return parent_dir_name
    except Exception:
        pass
    # 回退到原本的正則擷取方式
    path_str = str(path)
    import re
    pattern = rf"{dataset_name}_\d{{8}}_\d{{6}}"
    match = re.search(pattern, path_str)
    if match:
        return match.group(0)
    return None

def check_experiment_status(retrieval_path, args):
    """檢查實驗是否已完成，返回狀態和原因"""
    experiment_id = extract_experiment_id_from_path(retrieval_path, args.dataset_name)
    if experiment_id is None:
        experiment_id = Path(retrieval_path).parent.name
    
    result_dir = Path(f"./results/KGQA/{args.dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}/{experiment_id}")
    
    # 檢查最終結果檔案
    final_result = result_dir / f"{args.prompt_mode}-{args.llm_mode}-{args.frequency_penalty}-thres_{args.thres}-{args.split}-predictions.jsonl"
    
    # 檢查暫存檔案（可能中斷）
    temp_result = result_dir / f"{args.prompt_mode}-{args.llm_mode}-{args.frequency_penalty}-thres_{args.thres}-{args.split}-predictions-resume.jsonl"
    
    if final_result.exists():
        return "completed", "Final result file exists"
    elif temp_result.exists():
        # 檢查暫存檔案的行數，判斷是否部分完成
        try:
            with open(temp_result, 'r') as f:
                lines = sum(1 for _ in f)
            return "partial", f"Partial result exists ({lines} samples processed)"
        except:
            return "partial", "Partial result exists (unable to count lines)"
    else:
        return "pending", "No result files found"

def auto_run_all_retrieval_files(args):
    """自動運行指定目錄下所有的 retrieval_result.pth 檔案"""
    base_dir = f"/home/YX_thesis/retrieve/results/training/{args.dataset_name}"
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist!")
        return
    
    # 遞迴尋找所有 retrieval_result.pth 檔案
    pattern = os.path.join(base_dir, "**", "retrieval_result.pth")
    retrieval_files = glob.glob(pattern, recursive=True)
    
    if not retrieval_files:
        print(f"No retrieval_result.pth files found in {base_dir}")
        return
    
    print(f"Found {len(retrieval_files)} retrieval_result.pth files:")
    
    # 檢查每個檔案是否已經執行過
    valid_experiments = []
    skipped_experiments = []
    
    for file_path in retrieval_files:
        status, reason = check_experiment_status(file_path, args)
        
        if status == "completed" and not args.force_rerun:
            print(f"  ✓ {file_path} -> {reason}")
            skipped_experiments.append(file_path)
        elif status == "partial":
            print(f"  ⚠ {file_path} -> {reason}")
            if args.force_rerun:
                valid_experiments.append(file_path)
            else:
                # 自動繼續部分完成的實驗
                print(f"    -> Will resume from checkpoint")
                valid_experiments.append(file_path)
        else:
            print(f"  - {file_path} -> {reason}")
            valid_experiments.append(file_path)
    
    print(f"\nSummary:")
    print(f"  - Total files: {len(retrieval_files)}")
    print(f"  - Already completed: {len(skipped_experiments)}")
    print(f"  - Pending/Partial: {len(valid_experiments)}")
    
    if not valid_experiments:
        print("All experiments have been completed!")
        return
    
    # 只處理未完成的實驗
    print(f"\nStarting to process {len(valid_experiments)} pending experiments...")
    
    for i, retrieval_path in enumerate(valid_experiments, 1):
        print(f"\n{'='*60}")
        print(f"Processing experiment {i}/{len(valid_experiments)}")
        print(f"Retrieval file: {retrieval_path}")
        print(f"{'='*60}")
        
        try:
            # 創建新的參數副本，更新 score_dict_path
            exp_args = argparse.Namespace(**vars(args))
            exp_args.score_dict_path = retrieval_path
            
            # 運行單一實驗
            run_single_experiment(exp_args)
            
            print(f"✓ Successfully completed experiment: {retrieval_path}")
            
        except Exception as e:
            print(f"✗ Error processing experiment {retrieval_path}: {str(e)}")
            print("Continuing with next experiment...")
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed processing {len(valid_experiments)} pending experiments!")
    print(f"{'='*60}")

def run_single_experiment(args):
    """運行單一實驗的邏輯"""
    # 取得各參數值
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres
    reverse_order = getattr(args, "reverse_order", False)
    run_mode = getattr(args, "run_mode", "both")
    pred_file_arg = getattr(args, "pred_file", None)
    
    do_infer = run_mode in ("both", "infer")
    do_eval = run_mode in ("both", "eval")
    
    # 預測結果檔案路徑（RoG baseline 用）
    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    if not os.path.exists(pred_file_path):
        print(f"Warning: RoG baseline file not found: {pred_file_path}")
        print("Proceeding without RoG reasoning paths. Some prompt modes that require 'rog' will be unavailable.")
        pred_file_path = None
    
    # wandb run 名稱
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}"
    # 初始化 wandb，config 需轉成 dict
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=vars(args))
    
    # 決定 score_dict_path
    score_dict_path = None
    if do_infer:
        if args.score_dict_path is None:
            print("score_dict_path not been assigned")
            exit()
        else:
            score_dict_path = args.score_dict_path
            # 移除路徑前後空白避免 FileNotFoundError
            if isinstance(score_dict_path, str):
                score_dict_path = score_dict_path.strip()
    
    # 預測結果暫存檔案夾與檔案路徑（支援 resume）
    # 測試模式：使用 testing 目錄
    if getattr(args, "test_mode", False):
        raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/testing")
    else:
        raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    
    # 為本模型準備統一的 CSV 路徑（保持在模型根目錄）
    unified_csv_path = raw_pred_folder_path / "all_experiments_metrics.csv"
    
    # 提取實驗 ID
    experiment_id = None
    if do_infer:
        experiment_id = extract_experiment_id_from_path(score_dict_path, dataset_name)
        if experiment_id is None:
            # 如果無法從路徑提取，則使用時間戳作為備用
            experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Warning: Could not extract experiment ID from score_dict_path, using timestamp: {experiment_id}")
    elif do_eval and pred_file_arg is not None:
        # eval-only 且使用外部 predictions 檔案時，嘗試從檔案路徑推斷實驗 ID（僅用於 CSV 命名與日誌）
        try:
            experiment_id = extract_experiment_id_from_path(pred_file_arg, dataset_name)
        except Exception:
            experiment_id = None
    
    # 建立實驗 ID 資料夾（用於存放實驗特定檔案）
    experiment_folder_path = None
    if experiment_id is not None:
        experiment_folder_path = raw_pred_folder_path / experiment_id
        if experiment_folder_path.exists():
            print(f"使用現有的實驗資料夾: {experiment_folder_path}")
        else:
            experiment_folder_path.mkdir(parents=True, exist_ok=True)
            print(f"建立新的實驗資料夾: {experiment_folder_path}")
    
    # 建立實驗名稱（用於 CSV 記錄，格式與 aggregated_eval.csv 一致）
    # Include llm_mode in experiment name for clearer comparison across modes
    if experiment_id is not None:
        experiment_name = f"{experiment_id}-{prompt_mode}-{llm_mode}"
    else:
        experiment_name = f"{prompt_mode}-{llm_mode}"
    if reverse_order:
        experiment_name += "_rev_seq"
    
    # 測試模式：添加 _test_{k} 後綴
    if getattr(args, "test_mode", False):
        test_k = getattr(args, "test_k", 10)
        experiment_name += f"_test_{test_k}"
    
    # 實驗特定檔案路徑（存到實驗 ID 資料夾）
    raw_pred_file_path = None
    if do_infer:
        raw_pred_file_path = experiment_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions-resume.jsonl"
    
    # 在單檔模式也檢查是否已完成/部分完成，並處理 --force_rerun（僅在 inference 時）
    final_pred_file_path = None
    if experiment_folder_path is not None:
        final_pred_file_path = experiment_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions.jsonl"
    if do_infer and final_pred_file_path is not None:
        if final_pred_file_path.exists() and not args.force_rerun:
            print(f"Detected existing final result. Skip since --force_rerun not set: {final_pred_file_path}")
            return
        if args.force_rerun:
            # 強制重跑時，清理既有輸出
            try:
                if final_pred_file_path.exists():
                    print(f"--force_rerun: removing existing final result {final_pred_file_path}")
                    os.remove(final_pred_file_path)
            except Exception as e:
                print(f"Warning: failed to remove final file: {e}")
            try:
                if raw_pred_file_path is not None and raw_pred_file_path.exists():
                    print(f"--force_rerun: removing existing resume file {raw_pred_file_path}")
                    os.remove(raw_pred_file_path)
            except Exception as e:
                print(f"Warning: failed to remove resume file: {e}")
        else:
            # 非強制時，如有暫存檔會自動續跑（下方 load_checkpoint 會處理）
            if raw_pred_file_path is not None and raw_pred_file_path.exists():
                print(f"Found resume file. Will auto-resume from checkpoint: {raw_pred_file_path}")
    
    if do_infer:
        # 初始化 LLM
        llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty)
        # 取得資料
        data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)
        
        # 測試模式：對資料進行採樣
        if getattr(args, "test_mode", False):
            test_k = getattr(args, "test_k", 10)
            test_random = getattr(args, "test_random", False)
            
            original_data_size = len(data)
            if original_data_size > test_k:
                if test_random:
                    # 隨機採樣
                    random.seed(seed)  # 使用相同的seed確保可重現性
                    data = random.sample(data, test_k)
                    print(f"Test mode: Randomly sampled {test_k} samples from {original_data_size} total samples")
                else:
                    # 固定採樣（取前k個）
                    data = data[:test_k]
                    print(f"Test mode: Using first {test_k} samples from {original_data_size} total samples")
            else:
                print(f"Test mode: Dataset size ({original_data_size}) <= test_k ({test_k}), using all samples")
        
        # 取得 prompt
        sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
        print("Generating prompts...")
        # 產生每筆資料的 prompt
        data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)
    
        print("Starting inference...")
        # 初始化token計數器
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 取得已處理的資料數（支援 resume）
        start_idx = len(load_checkpoint(raw_pred_file_path))
        with open(raw_pred_file_path, "a") as pred_file:
            # 逐筆進行推論
            for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
                # 在第一次推理時顯示input token數量
                if idx == 0:
                    # 計算第一個樣本的input token數量（用於顯示）
                    sample_input_tokens = 0
                    if 'sys' in llm_mode:
                        sample_input_tokens += len(each_qa['sys_query'].split()) // 0.75 if 'sys_query' in each_qa else 0
                    if 'icl' in llm_mode:
                        sample_input_tokens += len(each_qa.get('user_query', '').split()) // 0.75
                    if 'user_query' in each_qa:
                        sample_input_tokens += len(each_qa['user_query'].split()) // 0.75
                    print(f"Estimated input tokens per sample: {int(sample_input_tokens)}")
                
                res, input_tokens, output_tokens = llm_inf_all(llm, each_qa, llm_mode, model_name)
                
                # 累計token數量
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # 移除不需要儲存的欄位，減少檔案大小（若不存在則忽略）
                for k in ["graph", "good_paths_rog", "good_triplets_rog", "scored_triplets"]:
                    each_qa.pop(k, None)

                # 儲存預測結果
                each_qa["prediction"] = res[0]
                save_checkpoint(pred_file, each_qa)
        
        # 在推理結束時顯示總token數量
        print(f"\n{'='*50}")
        print(f"Token Usage Summary:")
        print(f"Total Input Tokens: {total_input_tokens:,}")
        print(f"Total Output Tokens: {total_output_tokens:,}")
        print(f"Total Tokens: {total_input_tokens + total_output_tokens:,}")
        print(f"{'='*50}")
    
        # 處理完成後，將檔案重新命名（移除 -resume）
        final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
        os.rename(raw_pred_file_path, final_pred_file_path)
    
    # 僅在需要評估時執行評估；允許直接指定外部 predictions 檔案
    if do_eval:
        target_pred_path = None
        if do_infer:
            target_pred_path = final_pred_file_path
        else:
            if pred_file_arg is not None:
                target_pred_path = Path(pred_file_arg)
            elif experiment_folder_path is not None:
                candidate = experiment_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions.jsonl"
                target_pred_path = candidate
            else:
                print("Error: No predictions file provided for eval-only mode. Use --pred_file to specify the path.")
                return

        if not Path(target_pred_path).exists():
            print(f"Error: Predictions file not found for evaluation: {target_pred_path}")
            return

        # 進行評估，並將結果 append 到統一 CSV
        eval_all(target_pred_path, run, subset=True, split=split, experiment_name=experiment_name, csv_path=unified_csv_path, reverse_order=reverse_order)
        eval_all(target_pred_path, run, subset=False, split=split, experiment_name=experiment_name, csv_path=unified_csv_path, reverse_order=reverse_order)

# 主程式入口，負責整個流程的控制
def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    # meta-llama/Llama-3.2-3B, gpt-oss:20b, gpt-oss:120b, Qwen/Qwen3-8B
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    
    parser.add_argument("--auto_run_all", action="store_true", help="Automatically run all retrieval_result.pth files in training directory")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun even if results already exist")
    parser.add_argument("--run_mode", type=str, choices=["both", "infer", "eval"], default="both", help="Run only inference, only evaluation, or both")
    parser.add_argument("--pred_file", type=str, default=None, help="Path to an existing predictions.jsonl for eval-only mode")
    
    parser.add_argument("-rev", "--reverse_order", action="store_true", help="Reverse the order of triplets for evaluation")
    
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode for sampling limited data")
    parser.add_argument("--test_k", type=int, default=10, help="Number of samples to use in test mode (default: 10)")
    parser.add_argument("--test_random", action="store_true", help="Use random sampling in test mode (default: fixed samples)")

    args = parser.parse_args()
    
    # 如果啟用自動運行所有檔案
    if args.auto_run_all:
        auto_run_all_retrieval_files(args)
        return
    
    # 原有的單一實驗邏輯
    run_single_experiment(args)

# 程式進入點
if __name__ == "__main__":
    
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start reasoning {ts_start} ==========")
    
    main()

    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End reasoning {ts_end} ==========")
    print(f"Reasoning time: {end_time - start_time:.2f} seconds")

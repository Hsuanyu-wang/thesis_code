import os
import time
import openai
from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt
import tiktoken


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192 * 2, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0):
    # 檢查是否為 Ollama 的 gpt-oss 模型
    if "gpt-oss" in model_name:
        print(f"Using Ollama model: {model_name}")
        # 設定環境變數
        os.environ["OPENAI_BASE_URL"] = "http://192.168.63.184:11434/v1"
        os.environ["OPENAI_API_KEY"] = "ollama"
        
        # 使用 OpenAI 相容的 API
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens, frequency_penalty=frequency_penalty)
        return llm
    
    # 檢查是否為其他 GPT 模型（付費 API）
    elif "gpt" in model_name and "gpt-oss" not in model_name:
        print("[WARNING] Paid API models (like OpenAI GPT) are disabled on this system. Using a small local model instead.")
        # Default to a small model that fits in 4090 VRAM
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # 使用 vLLM 進行本地模型推理
    client = LLM(
        model=model_name, 
        tensor_parallel_size=tensor_parallel_size, 
        max_seq_len_to_capture=max_seq_len_to_capture,
        gpu_memory_utilization=0.8,  # 設置 GPU 內存利用率為 80%
        max_model_len=8192*2  # 限制最大模型長度
    )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                     frequency_penalty=frequency_penalty)
    llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    return llm


def get_outputs(outputs, model_name):
    if "gpt-oss" in model_name:
        # Ollama 模型使用 OpenAI 相容 API
        return outputs.choices[0].message.content
    elif "gpt" not in model_name:
        # vLLM 本地模型
        return outputs[0].outputs[0].text
    else:
        # 其他 GPT 模型
        return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name):
    res = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]
        # 計算system prompt的token數
        total_input_tokens += count_tokens(prompts['sys_query'], model_name)

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})
        # 計算ICL的token數
        total_input_tokens += count_tokens(icl_user_prompt, model_name)
        total_input_tokens += count_tokens(icl_ass_prompt, model_name)

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        # 計算user query的token數
        total_input_tokens += count_tokens(prompts['user_query'], model_name)
        
        outputs = get_outputs(llm(messages=conversation), model_name)
        # 計算output的token數
        total_output_tokens += count_tokens(outputs, model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        # 計算COT query的token數
        total_input_tokens += count_tokens(prompts['cot_query'], model_name)
        
        outputs = get_outputs(llm(messages=conversation), model_name)
        # 計算COT output的token數
        total_output_tokens += count_tokens(outputs, model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            # 計算COT query的token數
            total_input_tokens += count_tokens(prompts['cot_query'], model_name)
            
            outputs = get_outputs(llm(messages=conversation), model_name)
            # 計算COT output的token數
            total_output_tokens += count_tokens(outputs, model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res, total_input_tokens, total_output_tokens


# 全局緩存編碼器，避免重複初始化
_encoding_cache = {}

def count_tokens(text, model_name):
    """計算文本的token數量"""
    try:
        if 'gpt' in model_name:
            # 對於GPT模型，使用tiktoken
            # 使用緩存避免重複初始化編碼器
            if model_name not in _encoding_cache:
                _encoding_cache[model_name] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            encoding = _encoding_cache[model_name]
            return len(encoding.encode(text))
        else:
            # 對於其他模型，使用簡單的詞數估算（1 token ≈ 0.75 words）
            words = len(text.split())
            return int(words / 0.75)
    except:
        # 如果無法計算，使用簡單估算
        words = len(text.split())
        return int(words / 0.75)

def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt-oss' in model_name or ('gpt' in model_name and 'gpt-oss' not in model_name):
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)

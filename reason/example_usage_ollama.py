#!/usr/bin/env python3
"""
使用 Ollama gpt-oss:20b 模型的範例
"""

import os
import sys
from llm_utils import llm_init, llm_inf_all

def main():
    """示範如何使用 Ollama gpt-oss:20b 模型"""
    
    # 設定模型名稱
    model_name = "gpt-oss:20b"
    
    print("=" * 60)
    print("Ollama gpt-oss:20b 模型使用範例")
    print("=" * 60)
    
    # 初始化模型
    print(f"初始化模型: {model_name}")
    llm = llm_init(model_name)
    
    # 範例 1: 基本問答
    print("\n範例 1: 基本問答")
    qa_data = {
        'sys_query': '你是一個有用的助手，請簡潔地回答問題。',
        'user_query': '什麼是人工智慧？',
        'cot_query': '請詳細解釋人工智慧的發展歷程。'
    }
    
    result = llm_inf_all(llm, qa_data, 'sys_icl_dc', model_name)
    print(f"問題: {qa_data['user_query']}")
    print(f"回答: {result[0]}")
    
    # 範例 2: 知識圖譜問答
    print("\n範例 2: 知識圖譜問答")
    kg_qa_data = {
        'sys_query': '你是一個知識圖譜專家，請基於提供的三元組信息回答問題。',
        'user_query': '基於以下信息：蘋果是一種水果，蘋果富含維生素C。請問蘋果有什麼營養價值？',
        'cot_query': '請詳細分析蘋果的營養成分。'
    }
    
    result = llm_inf_all(llm, kg_qa_data, 'sys_icl_dc', model_name)
    print(f"問題: {kg_qa_data['user_query']}")
    print(f"回答: {result[0]}")
    
    print("\n✓ 範例執行完成！")

if __name__ == "__main__":
    main()


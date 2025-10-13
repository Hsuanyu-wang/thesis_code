#!/usr/bin/env python3
"""
Script to iterate through different llm_mode options for RAG KGQA experiments.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(llm_mode, score_dict_path, base_args=None):
    """Run a single experiment with the given llm_mode"""
    
    # Base command
    cmd = [
        "python", "main.py",
        "--force_rerun",
        "--reverse_order",
        "--llm_mode", llm_mode,
        "-p", score_dict_path
    ]
    
    # Add any additional base arguments
    if base_args:
        cmd.extend(base_args)
    
    print(f"\n{'='*80}")
    print(f"Running experiment with llm_mode: {llm_mode}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✓ Successfully completed {llm_mode} in {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✗ Failed to run {llm_mode} after {duration:.2f} seconds")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted {llm_mode}")
        return False

def main():
    """Main function to run all llm_mode experiments"""
    
    # Define the score_dict_path
    score_dict_path = "/home/YX_thesis/retrieve/results/training/webqsp/webqsp_Oct08-23:14:02_spcount/retrieval_result.pth"
    
    # Verify the path exists
    if not Path(score_dict_path).exists():
        print(f"Error: Score dict path does not exist: {score_dict_path}")
        sys.exit(1)
    
    # Define all llm_mode options to test
    llm_modes = [
        # Basic modes
        "sys",
        "sys_icl",
        "sys_dc",
        "sys_icl_dc",
        
        # COT modes
        "sys_sys_cot",
        "sys_icl_sys_cot",
        "sys_sys_cot_clear",
        "sys_icl_sys_cot_clear",
        
        # DC + COT combinations
        "sys_dc_sys_cot",
        "sys_icl_dc_sys_cot",
        "sys_dc_sys_cot_clear",
        "sys_icl_dc_sys_cot_clear",
    ]
    
    # Optional: Add any additional base arguments
    base_args = [
        # "--model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Default
        # "--prompt_mode", "scored_100",  # Default
        # "--split", "test",  # Default
        # "--temperature", "0",  # Default
        # "--frequency_penalty", "0.16",  # Default
    ]
    
    print(f"Starting experiments with {len(llm_modes)} different llm_mode options")
    print(f"Score dict path: {score_dict_path}")
    print(f"Base arguments: {base_args if base_args else 'None (using defaults)'}")
    
    # Track results
    successful = []
    failed = []
    
    # Run each experiment
    for i, llm_mode in enumerate(llm_modes, 1):
        print(f"\n{'#'*100}")
        print(f"Experiment {i}/{len(llm_modes)}: {llm_mode}")
        print(f"{'#'*100}")
        
        success = run_experiment(llm_mode, score_dict_path, base_args)
        
        if success:
            successful.append(llm_mode)
        else:
            failed.append(llm_mode)
        
        # Optional: Add a small delay between experiments
        time.sleep(2)
    
    # Print summary
    print(f"\n{'='*100}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    print(f"Total experiments: {len(llm_modes)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successful llm_modes:")
        for mode in successful:
            print(f"  - {mode}")
    
    if failed:
        print(f"\n✗ Failed llm_modes:")
        for mode in failed:
            print(f"  - {mode}")
    
    print(f"\n{'='*100}")

if __name__ == "__main__":
    main()

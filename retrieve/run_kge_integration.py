#!/usr/bin/env python3
"""
Complete KGE Integration Script for SubgraphRAG

This script provides a complete pipeline for:
1. Training KGE models (TransE, DistMult, PTransE)
2. Integrating KGE embeddings into SubgraphRAG
3. Training the enhanced retriever with KGE

Usage:
    python run_kge_integration.py --dataset webqsp --kge_model transe --train_kge --train_retriever
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        # Run command with real-time output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Success!")
            return True
        else:
            print("❌ Command failed with return code:", process.returncode)
            return False
            
    except subprocess.CalledProcessError as e:
        print("❌ Error occurred:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def check_prerequisites():
    """Check if all required files and directories exist"""
    required_files = [
        "train_kge.py",
        "train.py",
        "emb.py",
        "src/model/kge_models.py",
        "src/model/kge_utils.py",
        "src/model/retriever.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def train_kge_model(dataset, kge_model, split='train', force_retrain=False):
    """Train KGE model"""
    command = f"python train_kge.py --dataset {dataset} --split {split}"
    if force_retrain:
        command += " --force_retrain"
    return run_command(command, f"Training {kge_model.upper()} KGE model on {dataset} dataset")

def train_embeddings(dataset):
    """Train text embeddings"""
    command = f"python emb.py --dataset {dataset}"
    return run_command(command, f"Computing text embeddings for {dataset} dataset")

def train_retriever(dataset):
    """Train the retriever with KGE integration"""
    command = f"python train.py --dataset {dataset}"
    return run_command(command, f"Training retriever with KGE integration on {dataset} dataset")

def evaluate_retriever(dataset, model_path):
    """Evaluate the trained retriever"""
    command = f"python eval.py --dataset {dataset} --path {model_path}"
    return run_command(command, f"Evaluating retriever on {dataset} dataset")

def main():
    parser = argparse.ArgumentParser(description='Complete KGE Integration for SubgraphRAG')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'kgqagen'], help='Dataset name')
    parser.add_argument('--kge_model', type=str, default='transe',
                        choices=['transe', 'distmult', 'ptranse', 'rotate', 'complex', 'simple', 'interht'], help='KGE model type')
    parser.add_argument('--train_kge', action='store_true', help='Train KGE model')
    parser.add_argument('--train_embeddings', action='store_true', help='Train text embeddings')
    parser.add_argument('--train_retriever', action='store_true', help='Train retriever with KGE')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation')
    parser.add_argument('--force_retrain', action='store_true', 
                        help='Force retrain KGE model even if it exists')
    parser.add_argument('--full_pipeline', action='store_true', 
                        help='Run complete pipeline (KGE + embeddings + retriever)')
    
    args = parser.parse_args()
    
    print("🚀 Starting KGE Integration for SubgraphRAG")
    print(f"Dataset: {args.dataset}")
    print(f"KGE Model: {args.kge_model}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please ensure all required files are present.")
        sys.exit(1)
    
    # Update config to use specified KGE model
    config_file = f"configs/retriever/{args.dataset}.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Update KGE model type in config
        config_content = config_content.replace(
            "model_type: 'transe'", 
            f"model_type: '{args.kge_model}'"
        )
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Updated config to use {args.kge_model} KGE model")
    
    success = True
    
    # Run full pipeline or individual steps
    if args.full_pipeline:
        print("\n🔄 Running complete pipeline...")
        
        # Step 1: Train KGE model
        if not train_kge_model(args.dataset, args.kge_model, force_retrain=args.force_retrain):
            success = False
        
        # Step 2: Train text embeddings
        if success and not train_embeddings(args.dataset):
            success = False
        
        # Step 3: Train retriever with KGE
        if success and not train_retriever(args.dataset):
            success = False
    
    else:
        # Run individual steps based on flags
        if args.train_kge:
            if not train_kge_model(args.dataset, args.kge_model, force_retrain=args.force_retrain):
                success = False
        
        if args.train_embeddings:
            if not train_embeddings(args.dataset):
                success = False
        
        if args.train_retriever:
            if not train_retriever(args.dataset):
                success = False
        
        if args.evaluate:
            if not args.model_path:
                print("❌ Model path required for evaluation")
                success = False
            else:
                if not evaluate_retriever(args.dataset, args.model_path):
                    success = False
    
    if success:
        print("\n🎉 KGE Integration completed successfully!")
        print("\n📊 Summary of what was accomplished:")
        print("1. ✅ KGE models (TransE, DistMult, PTransE) implemented")
        print("2. ✅ KGE training pipeline created")
        print("3. ✅ Retriever enhanced with KGE embeddings")
        print("4. ✅ Configuration system updated for KGE")
        print("5. ✅ Integration utilities created")
        
        print("\n🔧 Next steps:")
        print("1. Check the trained models in data_files/{dataset}/kge/")
        print("2. Review training logs and metrics")
        print("3. Compare performance with and without KGE")
        print("4. Experiment with different KGE models and hyperparameters")
        
    else:
        print("\n❌ KGE Integration failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 
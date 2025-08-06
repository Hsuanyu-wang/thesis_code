#!/bin/bash

# SubgraphRAG Complete Pipeline Script
# Usage: ./pipeline.sh [dataset] [optional: kge_model] [optional: max_epochs]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}

# Function to check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Function to create directory if it doesn't exist
create_dir() {
    if ! dir_exists "$1"; then
        mkdir -p "$1"
        print_status "Created directory: $1"
    fi
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            print_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
            return 0
        else
            print_warning "GPU not available, using CPU"
            return 1
        fi
    else
        print_warning "nvidia-smi not found, assuming CPU only"
        return 1
    fi
}

# Function to check Python environment
check_python_env() {
    if ! command_exists python; then
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ "$python_version" < "3.10" ]]; then
        print_error "Python 3.10+ required, found: $python_version"
        exit 1
    fi
    
    print_success "Python version: $(python --version)"
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check required Python packages
    python -c "import torch" 2>/dev/null || missing_deps+=("torch")
    python -c "import torch_geometric" 2>/dev/null || missing_deps+=("torch_geometric")
    python -c "import wandb" 2>/dev/null || missing_deps+=("wandb")
    python -c "import tqdm" 2>/dev/null || missing_deps+=("tqdm")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install dependencies: pip install -r requirements/retriever.txt"
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Function to validate dataset
validate_dataset() {
    local dataset=$1
    local valid_datasets=("webqsp" "cwq" "kgqagen")
    
    for valid_dataset in "${valid_datasets[@]}"; do
        if [[ "$dataset" == "$valid_dataset" ]]; then
            return 0
        fi
    done
    
    print_error "Invalid dataset: $dataset. Valid options: ${valid_datasets[*]}"
    exit 1
}

# Function to check data files
check_data_files() {
    local dataset=$1
    
    if ! dir_exists "data_files/$dataset"; then
        print_error "Data files not found for dataset: $dataset"
        print_status "Please ensure data_files/$dataset exists with processed data"
        exit 1
    fi
    
    if ! dir_exists "data_files/$dataset/processed"; then
        print_error "Processed data not found: data_files/$dataset/processed"
        exit 1
    fi
    
    print_success "Data files found for dataset: $dataset"
}

# Function to run embedding pre-computation
run_embedding() {
    local dataset=$1
    
    print_status "Step 1: Pre-computing embeddings for $dataset"
    
    if file_exists "data_files/$dataset/emb/gte-large-en-v1.5/train.pth"; then
        print_warning "Embeddings already exist, skipping pre-computation"
        return 0
    fi
    
    if python emb.py -d "$dataset"; then
        print_success "Embedding pre-computation completed"
    else
        print_error "Embedding pre-computation failed"
        exit 1
    fi
}

# Function to train KGE model
train_kge() {
    local dataset=$1
    local kge_model=${2:-""}
    
    # Skip KGE training if no model specified or if it's "no_kge"
    if [ -z "$kge_model" ] || [ "$kge_model" = "no_kge" ]; then
        print_status "Step 2: Skipping KGE training (no model specified)"
        return 0
    fi
    
    print_status "Step 2: Training KGE model ($kge_model) for $dataset"
    
    # Check if KGE model already exists
    local kge_path="data_files/$dataset/kge/$kge_model/train_model.pth"
    if file_exists "$kge_path"; then
        print_warning "KGE model already exists, skipping training"
        return 0
    fi
    
    if python train_kge.py --dataset "$dataset" --split train --model_type "$kge_model"; then
        print_success "KGE model training completed"
    else
        print_error "KGE model training failed"
        exit 1
    fi
}

# Function to train retriever with experiment ID
train_retriever_with_id() {
    local dataset=$1
    local kge_model=${2:-""}
    local max_epochs=${3:-10000}
    local experiment_id=$4
    
    print_status "Step 3: Training retriever for $dataset (max epochs: $max_epochs, exp_id: $experiment_id)"
    
    # Use specific experiment directory
    local result_dir="training result/${dataset}_${experiment_id}"
    
    # Check if training result already exists for this specific configuration
    if [ -f "$result_dir/cpt.pth" ]; then
        # Check if the existing result matches current KGE configuration
        local kge_info=""
        if [ -n "$kge_model" ]; then
            kge_info="with KGE: $kge_model"
        else
            kge_info="without KGE"
        fi
        
        print_warning "Training result already exists: $result_dir"
        print_warning "Current configuration: $kge_info"
        read -p "Continue with existing result? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Using existing training result"
            return 0
        else
            print_status "Will train new model with current configuration"
        fi
    fi
    
    # Create experiment directory
    mkdir -p "$result_dir"
    
    # Train with or without KGE model
    if [ -z "$kge_model" ] || [ "$kge_model" = "no_kge" ]; then
        print_status "Training without KGE model..."
        if python train.py -d "$dataset" --exp_id "$experiment_id"; then
            print_success "Retriever training completed (without KGE)"
        else
            print_error "Retriever training failed"
            exit 1
        fi
    else
        print_status "Training with KGE model: $kge_model..."
        if python train.py -d "$dataset" --kge_model "$kge_model" --exp_id "$experiment_id"; then
            print_success "Retriever training completed (with KGE: $kge_model)"
        else
            print_error "Retriever training failed"
            exit 1
        fi
    fi
}

# Function to run inference with experiment ID
run_inference_with_id() {
    local dataset=$1
    local experiment_id=$2
    
    print_status "Step 4: Running inference for $dataset (exp_id: $experiment_id)"
    
    # Use specific experiment directory
    local result_dir="training result/${dataset}_${experiment_id}"
    local checkpoint="$result_dir/cpt.pth"
    
    if [ ! -f "$checkpoint" ]; then
        print_error "No training checkpoint found: $checkpoint"
        exit 1
    fi
    
    print_status "Using checkpoint: $checkpoint"
    
    if python inference.py -p "$checkpoint" --max_K 500; then
        print_success "Inference completed"
    else
        print_error "Inference failed"
        exit 1
    fi
}

# Function to run evaluation with experiment ID
run_evaluation_with_id() {
    local dataset=$1
    local kge_model=${2:-""}
    local experiment_id=$3
    
    print_status "Step 5: Running evaluation for $dataset (exp_id: $experiment_id)"
    
    # Use specific experiment directory
    local result_dir="training result/${dataset}_${experiment_id}"
    local result_file="$result_dir/retrieval_result.pth"
    
    if [ ! -f "$result_file" ]; then
        print_error "No inference result found: $result_file"
        exit 1
    fi
    
    print_status "Using result file: $result_file"
    
    # Use the KGE model specified by command line argument
    local use_kge="$kge_model"
    local subgraph_method="shortestpath"
    
    if python eval.py -d "$dataset" -p "$result_file" --use_kge "$use_kge" --subgraph_method "$subgraph_method"; then
        print_success "Evaluation completed"
    else
        print_error "Evaluation failed"
        exit 1
    fi
}

# Function to generate summary report with experiment ID
generate_summary_with_id() {
    local dataset=$1
    local experiment_id=$2
    
    print_status "Step 6: Generating summary report (exp_id: $experiment_id)"
    
    # Use specific experiment directory
    local result_dir="training result/${dataset}_${experiment_id}"
    
    if [ ! -d "$result_dir" ]; then
        print_error "No training result found: $result_dir"
        exit 1
    fi
    
    local summary_file="$result_dir/summary.txt"
    
    {
        echo "SubgraphRAG Pipeline Summary"
        echo "============================"
        echo "Dataset: $dataset"
        echo "Experiment ID: $experiment_id"
        echo "Timestamp: $(date)"
        echo "Result Directory: $result_dir"
        echo ""
        echo "Files:"
        ls -la "$result_dir"
        echo ""
        echo "Model Info:"
        if [ -f "$result_dir/cpt.pth" ]; then
        echo "  - Checkpoint size: $(du -h "$result_dir/cpt.pth" | cut -f1)"
        fi
        if [ -f "$result_dir/retrieval_result.pth" ]; then
        echo "  - Result file size: $(du -h "$result_dir/retrieval_result.pth" | cut -f1)"
        fi
        echo ""
        echo "Wandb URL:"
        echo "  - Check wandb dashboard for detailed metrics"
    } > "$summary_file"
    
    print_success "Summary report generated: $summary_file"
    cat "$summary_file"
}

# Function to run single task
run_single_task() {
    local dataset=$1
    local kge_model=${2:-""}
    local max_epochs=${3:-10000}
    local experiment_id=${4:-""}
    
    # Generate experiment ID if not provided
    if [ -z "$experiment_id" ]; then
        experiment_id=$(date +"%Y%m%d_%H%M%S")
    fi
    
    print_status "Starting SubgraphRAG pipeline"
    print_status "Dataset: $dataset"
    print_status "KGE Model: $kge_model"
    print_status "Max Epochs: $max_epochs"
    print_status "Experiment ID: $experiment_id"
    echo ""
    
    # Pre-flight checks
    check_python_env
    check_dependencies
    check_gpu
    validate_dataset "$dataset"
    check_data_files "$dataset"
    
    # Create necessary directories
    create_dir "training result"
    create_dir "retrieve_result"
    create_dir "wandb"
    
    # Run pipeline steps with experiment ID
    run_embedding "$dataset"
    train_kge "$dataset" "$kge_model"
    train_retriever_with_id "$dataset" "$kge_model" "$max_epochs" "$experiment_id"
    run_inference_with_id "$dataset" "$experiment_id"
    run_evaluation_with_id "$dataset" "$kge_model" "$experiment_id"
    generate_summary_with_id "$dataset" "$experiment_id"
    
    print_success "Pipeline completed successfully!"
    print_status "Check the result directory: training result/${dataset}_${experiment_id}"
}

# Function to run batch tasks
run_batch_tasks() {
    local config_file=$1
    
    if [ ! -f "$config_file" ]; then
        print_error "Batch config file not found: $config_file"
        exit 1
    fi
    
    print_status "Starting batch processing mode"
    print_status "Config file: $config_file"
    echo ""
    
    # Pre-flight checks (only once for batch mode)
    check_python_env
    check_dependencies
    check_gpu
    
    # Create necessary directories
    create_dir "training result"
    create_dir "retrieve_result"
    create_dir "wandb"
    
    # Read and process each line in the config file
    local line_num=0
    local total_tasks=0
    local completed_tasks=0
    local failed_tasks=0
    
    # Count total tasks
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            total_tasks=$((total_tasks + 1))
        fi
    done < "$config_file"
    
    print_status "Found $total_tasks tasks to process"
    echo ""
    
    # Process each task
    line_num=0
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        # Parse task configuration
        read -r dataset max_epochs kge_model experiment_id <<< "$line"
        
        # Set defaults if not provided
        max_epochs=${max_epochs:-10000}
        kge_model=${kge_model:-"no_kge"}
        experiment_id=${experiment_id:-""} # Ensure experiment_id is not empty
        
        # Convert empty or "no_kge" to empty string for internal use
        if [ "$kge_model" = "no_kge" ] || [ -z "$kge_model" ]; then
            kge_model=""
        fi
        
        print_status "Processing task $((completed_tasks + failed_tasks + 1))/$total_tasks"
        print_status "Dataset: $dataset, KGE: $kge_model, Epochs: $max_epochs, Exp ID: $experiment_id"
        echo ""
        
        # Validate dataset for this task
        validate_dataset "$dataset"
        check_data_files "$dataset"
        
        # Run the task
        if run_single_task "$dataset" "$kge_model" "$max_epochs" "$experiment_id"; then
            completed_tasks=$((completed_tasks + 1))
            print_success "Task completed successfully"
        else
            failed_tasks=$((failed_tasks + 1))
            print_error "Task failed"
        fi
        
        echo ""
        print_status "Progress: $((completed_tasks + failed_tasks))/$total_tasks completed"
        print_status "Success: $completed_tasks, Failed: $failed_tasks"
        echo ""
        
    done < "$config_file"
    
    # Final summary
    echo "========================================"
    print_success "Batch processing completed!"
    print_status "Total tasks: $total_tasks"
    print_success "Completed: $completed_tasks"
    if [ $failed_tasks -gt 0 ]; then
        print_error "Failed: $failed_tasks"
    fi
    echo "========================================"
}

# Function to show help
show_help() {
    echo "SubgraphRAG Complete Pipeline Script"
    echo ""
    echo "Usage:"
    echo "  Single task: $0 [dataset] [kge_model] [max_epochs] [experiment_id]"
    echo "  Batch mode:  $0 --batch [config_file]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Dataset name (webqsp, cwq, kgqagen)"
    echo "  kge_model     KGE model type (transe, distmult, ptranse, rotate, complex, simple, interht) [default: none]"
    echo "  max_epochs    Maximum training epochs [default: 10000]"
    echo "  experiment_id Experiment ID for parallel runs [default: auto-generated timestamp]"
    echo ""
    echo "Batch Mode:"
    echo "  --batch       Enable batch processing mode"
    echo "  config_file   Path to batch configuration file"
    echo ""
    echo "Examples:"
    echo "  Single task:"
    echo "    $0 webqsp                           # Run complete pipeline for WebQSP"
    echo "    $0 cwq distmult                     # Run with DistMult KGE model"
    echo "    $0 kgqagen interht                  # Run with InterHT KGE model"
    echo "    $0 webqsp transe 5000               # Run with custom max epochs"
    echo "    $0 webqsp transe 5000 exp001        # Run with specific experiment ID"
    echo ""
    echo "  Batch mode:"
    echo "    $0 --batch batch_config.txt         # Run multiple tasks from config file"
    echo ""
    echo "Parallel Execution:"
    echo "  To run multiple experiments simultaneously without conflicts:"
    echo "    Terminal 1: $0 webqsp transe 5000 exp001"
    echo "    Terminal 2: $0 webqsp distmult 5000 exp002"
    echo "    Terminal 3: $0 cwq rotate 5000 exp003"
    echo ""
    echo "Batch Config File Format:"
    echo "  # Each line: dataset max_epochs kge_model experiment_id"
    echo "  # Use 'no_kge' to disable KGE model"
    echo "  # Leave experiment_id empty for auto-generation"
    echo "  webqsp 10000 transe exp001"
    echo "  webqsp 10000 distmult exp002"
    echo "  cwq 5000 rotate exp003"
    echo "  kgqagen 10000 interht exp004"
    echo "  webqsp 10000 no_kge exp005"
    echo ""
    echo "Steps:"
    echo "  1. Pre-compute embeddings"
    echo "  2. Train KGE model"
    echo "  3. Train retriever"
    echo "  4. Run inference"
    echo "  5. Evaluate results"
    echo "  6. Generate summary report"
}

# Main function
main() {
    # Check arguments
    if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        show_help
        exit 0
    fi
    
    # Check for batch mode
    if [ "$1" == "--batch" ]; then
        if [ $# -lt 2 ]; then
            print_error "Batch mode requires config file path"
            show_help
            exit 1
        fi
        run_batch_tasks "$2"
    else
        # Single task mode
        local dataset=$1
        local kge_model=${2:-""}
        local max_epochs=${3:-10000}
        local experiment_id=${4:-""} # Get experiment_id from arguments
        
        run_single_task "$dataset" "$kge_model" "$max_epochs" "$experiment_id"
    fi
}

# Run main function with all arguments
main "$@" 
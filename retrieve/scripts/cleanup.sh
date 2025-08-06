#!/bin/bash

# =============================================================================
# SubgraphRAG Cleanup Script
# 清理和整理檔案結構
# =============================================================================

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日誌函數
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 顯示幫助信息
show_help() {
    echo -e "${BLUE}SubgraphRAG Cleanup Script${NC}"
    echo ""
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  --backup_old_readme      備份舊的 README 檔案"
    echo "  --remove_old_experiments 移除舊的實驗結果"
    echo "  --organize_files          整理檔案結構"
    echo "  --clean_wandb            清理 wandb 日誌"
    echo "  --all                    執行所有清理操作"
    echo "  --help                   顯示此幫助信息"
    echo ""
    echo "範例:"
    echo "  $0 --backup_old_readme"
    echo "  $0 --all"
    echo ""
}

# 備份舊的 README 檔案
backup_old_readme() {
    log_step "備份舊的 README 檔案..."
    
    # 創建備份目錄
    mkdir -p backup_readme
    
    # 備份舊的 README 檔案
    if [[ -f "README_V2.md" ]]; then
        mv README_V2.md backup_readme/
        log_info "已備份 README_V2.md"
    fi
    
    if [[ -f "KGE_INTEGRATION_README.md" ]]; then
        mv KGE_INTEGRATION_README.md backup_readme/
        log_info "已備份 KGE_INTEGRATION_README.md"
    fi
    
    log_info "README 檔案備份完成"
}

# 移除舊的實驗結果
remove_old_experiments() {
    log_step "移除舊的實驗結果..."
    
    # 移除舊的實驗資料夾（保留最新的）
    for dataset in webqsp cwq; do
        # 找到該資料集的所有實驗資料夾
        exp_dirs=($(ls -td "${dataset}_"* 2>/dev/null || true))
        
        if [[ ${#exp_dirs[@]} -gt 1 ]]; then
            # 保留最新的，移除其他的
            for ((i=1; i<${#exp_dirs[@]}; i++)); do
                rm -rf "${exp_dirs[$i]}"
                log_info "已移除舊實驗: ${exp_dirs[$i]}"
            done
        fi
    done
    
    log_info "舊實驗結果清理完成"
}

# 整理檔案結構
organize_files() {
    log_step "整理檔案結構..."
    
    # 創建必要的目錄
    mkdir -p backup_readme
    mkdir -p docs
    mkdir -p scripts
    
    # 移動文檔檔案到 docs 目錄
    if [[ -f "FILES_DESCRIPTION.md" ]]; then
        mv FILES_DESCRIPTION.md docs/
        log_info "已移動 FILES_DESCRIPTION.md 到 docs/"
    fi
    
    if [[ -f "KGE_SCORE_INTEGRATION_SUMMARY.md" ]]; then
        mv KGE_SCORE_INTEGRATION_SUMMARY.md docs/
        log_info "已移動 KGE_SCORE_INTEGRATION_SUMMARY.md 到 docs/"
    fi
    
    # 移動腳本檔案到 scripts 目錄
    if [[ -f "cleanup.sh" ]]; then
        mv cleanup.sh scripts/
        log_info "已移動 cleanup.sh 到 scripts/"
    fi
    
    # 整理測試檔案
    mkdir -p tests
    for test_file in test_*.py demo_*.py; do
        if [[ -f "$test_file" ]]; then
            mv "$test_file" tests/
            log_info "已移動 $test_file 到 tests/"
        fi
    done
    
    log_info "檔案結構整理完成"
}

# 清理 wandb 日誌
clean_wandb() {
    log_step "清理 wandb 日誌..."
    
    if [[ -d "wandb" ]]; then
        # 保留最新的 5 個實驗
        cd wandb
        exp_dirs=($(ls -td */ 2>/dev/null | head -5))
        
        # 移除舊的實驗
        all_dirs=($(ls -td */ 2>/dev/null || true))
        for dir in "${all_dirs[@]}"; do
            if [[ ! " ${exp_dirs[@]} " =~ " ${dir} " ]]; then
                rm -rf "$dir"
                log_info "已移除舊 wandb 實驗: $dir"
            fi
        done
        cd ..
    fi
    
    log_info "wandb 日誌清理完成"
}

# 創建新的目錄結構
create_directory_structure() {
    log_step "創建標準目錄結構..."
    
    # 創建標準目錄
    mkdir -p {docs,scripts,tests,backup_readme}
    
    # 創建 .gitignore
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
wandb/
training result/
retrieve_result/
*.pth
*.pt
*.ckpt

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
EOF
    
    log_info "標準目錄結構創建完成"
}

# 主函數
main() {
    local backup_readme=false
    local remove_old_experiments=false
    local organize_files=false
    local clean_wandb=false
    
    # 解析命令行參數
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup_old_readme)
                backup_readme=true
                shift
                ;;
            --remove_old_experiments)
                remove_old_experiments=true
                shift
                ;;
            --organize_files)
                organize_files=true
                shift
                ;;
            --clean_wandb)
                clean_wandb=true
                shift
                ;;
            --all)
                backup_readme=true
                remove_old_experiments=true
                organize_files=true
                clean_wandb=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知參數: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果沒有指定任何操作，顯示幫助
    if [[ "$backup_readme" == "false" && "$remove_old_experiments" == "false" && "$organize_files" == "false" && "$clean_wandb" == "false" ]]; then
        show_help
        exit 0
    fi
    
    log_info "開始 SubgraphRAG 清理流程"
    
    # 創建標準目錄結構
    create_directory_structure
    
    # 執行指定的操作
    if [[ "$backup_readme" == "true" ]]; then
        backup_old_readme
    fi
    
    if [[ "$remove_old_experiments" == "true" ]]; then
        remove_old_experiments
    fi
    
    if [[ "$organize_files" == "true" ]]; then
        organize_files
    fi
    
    if [[ "$clean_wandb" == "true" ]]; then
        clean_wandb
    fi
    
    log_info "🎉 清理流程完成！"
    
    # 顯示最終的目錄結構
    log_step "最終目錄結構:"
    tree -L 2 -I '__pycache__|*.pyc|wandb|training result|retrieve_result' 2>/dev/null || find . -maxdepth 2 -type d | grep -v '__pycache__' | sort
}

# 執行主函數
main "$@" 
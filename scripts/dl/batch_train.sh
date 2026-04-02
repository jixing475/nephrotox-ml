#!/bin/bash

# ==============================================================================
# 批量训练脚本: 深度学习模型训练
# ==============================================================================
# 
# 支持三个后端:
#   - DeepChem: GCN
#   - DGLlife: GCN, GAT, Weave, AttentiveFP
#   - Chemprop: D-MPNN (支持特征融合)
#
# 工作流程: 先调优（tune）再评估（evaluate）
# 每个步骤独立执行，失败不影响下一个
#
# 使用示例:
#   ./batch_train.sh                                    # 默认 GCN_DeepChem_Graph
#   CONFIG=DMPNN_Chemprop_Graph ./batch_train.sh        # Chemprop D-MPNN
#   CONFIG=GAT_DGLlife_Graph CLASS_WEIGHTS=true ./batch_train.sh
#   BATCH_MODE=all ./batch_train.sh                     # 训练所有模型
#
# ==============================================================================

# 注意：不使用 set -e，以便每个步骤的失败不影响下一个

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==============================================================================
# 配置参数 - 可通过环境变量覆盖
# ==============================================================================
# 默认值从 src/config.py 动态读取，确保单一数据源
# ==============================================================================

# 使用Python从config.py读取默认值的辅助函数
get_config_value() {
    local var_name=$1
    local default=$2
    .venv/bin/python -c "from src import config; print(getattr(config, '$var_name', $default))" 2>/dev/null || echo "$default"
}

# 检查.venv是否存在，如果存在则从config.py读取默认值
if [ -f ".venv/bin/python" ]; then
    DEFAULT_CONFIG=$(get_config_value "DEFAULT_CONFIG_KEY" "GCN_DeepChem_Graph")
    DEFAULT_OPTUNA_TRIALS=$(get_config_value "OPTUNA_N_TRIALS" 100)
    DEFAULT_N_SEEDS=$(.venv/bin/python -c "from src import config; print(len(config.RANDOM_SEEDS))" 2>/dev/null || echo 10)
    DEFAULT_PATIENCE=$(.venv/bin/python -c "from src import config; print(config.TRAINING_PARAMS.get('early_stopping_patience', 25))" 2>/dev/null || echo 25)
    DEFAULT_METRIC=$(.venv/bin/python -c "from src import config; print(config.TRAINING_PARAMS.get('early_stopping_metric', 'roc_auc'))" 2>/dev/null || echo "roc_auc")
else
    # 如果虚拟环境不存在，使用硬编码默认值（应与config.py保持一致）
    DEFAULT_CONFIG="GCN_DeepChem_Graph"
    DEFAULT_OPTUNA_TRIALS=100
    DEFAULT_N_SEEDS=10
    DEFAULT_PATIENCE=25
    DEFAULT_METRIC="roc_auc"
fi

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"                        # 从config.py读取
OPTUNA_TRIALS="${OPTUNA_TRIALS:-$DEFAULT_OPTUNA_TRIALS}"   # 从config.py读取
PATIENCE="${PATIENCE:-$DEFAULT_PATIENCE}"                  # 从config.py读取  
METRIC="${METRIC:-$DEFAULT_METRIC}"                        # 从config.py读取
N_SEEDS="${N_SEEDS:-$DEFAULT_N_SEEDS}"                     # 从config.py读取
N_GPUS="${N_GPUS:-}"                             # GPU数量（空=使用所有可用GPU）
PARALLEL_TUNE="${PARALLEL_TUNE:-false}"          # 是否使用多GPU并行调优（默认关闭，TPE采样器在串行模式下效果更好）
BATCH_MODE="${BATCH_MODE:-single}"               # 批量模式: single | all_dgllife | all_chemprop | all
CLASS_WEIGHTS="${CLASS_WEIGHTS:-false}"          # 是否使用类别加权损失
OPTIMIZE_THRESHOLD="${OPTIMIZE_THRESHOLD:-false}" # 是否优化分类阈值
FORCE_CPU="${FORCE_CPU:-false}"                  # 是否强制使用CPU
SKIP_TUNE="${SKIP_TUNE:-false}"                  # 是否跳过调优

# ==============================================================================
# 模型配置列表
# ==============================================================================

# DeepChem 模型列表
DEEPCHEM_CONFIGS=(
    "GCN_DeepChem_Graph"
)

# DGLlife 模型列表
DGLLIFE_CONFIGS=(
    "GCN_DGLlife_Graph"
    "GAT_DGLlife_Graph"
    "Weave_DGLlife_Graph"
    "AttentiveFP_DGLlife_Graph"
)

# Chemprop 模型列表（D-MPNN）
CHEMPROP_CONFIGS=(
    "DMPNN_Chemprop_Graph"
    "DMPNN_Chemprop_Graph+RDKit"
    "DMPNN_Chemprop_Graph+ChemoPy2d"
)

# 所有模型
ALL_CONFIGS=(
    "${DEEPCHEM_CONFIGS[@]}"
    "${DGLLIFE_CONFIGS[@]}"
    "${CHEMPROP_CONFIGS[@]}"
)

# 创建日志文件（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="batch_training_${CONFIG}_${TIMESTAMP}.log"

# 统计变量
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# 记录开始时间
START_TIME=$(date +%s)

# 打印带颜色的消息并记录到日志
log_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOG_FILE"
}

# 检测后端类型
get_backend() {
    local config=$1
    case $config in
        *_DeepChem_*)
            echo "deepchem"
            ;;
        *_DGLlife_*)
            echo "dgllife"
            ;;
        *_Chemprop_*)
            echo "chemprop"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# 检测是否使用特征融合
is_fusion_mode() {
    local config=$1
    case $config in
        *+*)
            echo "true"
            ;;
        *)
            echo "false"
            ;;
    esac
}

# 运行调优 + 评估
run_training() {
    local current_config=$1
    
    log_message "$BLUE" "\n=========================================="
    log_message "$BLUE" "开始处理: $current_config"
    log_message "$BLUE" "=========================================="
    
    # 检测后端和特征类型
    local backend=$(get_backend "$current_config")
    local fusion=$(is_fusion_mode "$current_config")
    
    log_message "$CYAN" "后端: $backend | 融合特征: $fusion"
    
    # 检查是否已完成（可选：跳过已完成的组合）
    # 取消注释以下代码以启用跳过功能
    # if [ -f "output/$current_config/cv_summary.csv" ] && [ -f "output/$current_config/best_params.json" ]; then
    #     log_message "$YELLOW" "⚠  跳过 $current_config（已完成）"
    #     SKIP_COUNT=$((SKIP_COUNT + 1))
    #     return 0
    # fi
    
    # 检查虚拟环境是否存在
    if [ ! -d ".venv" ]; then
        log_message "$YELLOW" "虚拟环境不存在，正在创建..."
        uv venv
        uv pip install -e .
        # 安装 CUDA 版本的 DGL
        uv pip install "dgl>=1.1.0,<2.0" -f https://data.dgl.ai/wheels/cu118/repo.html
    fi
    
    # 使用虚拟环境中的 Python
    PYTHON_CMD=".venv/bin/python"
    if [ ! -f "$PYTHON_CMD" ]; then
        log_message "$RED" "错误: 虚拟环境中的 Python 不存在"
        return 1
    fi
    
    # 设置 CUDA 环境（Chemprop 现在可以支持 GPU）
    local cuda_env=""
    if [ "$FORCE_CPU" = "true" ]; then
        cuda_env="CUDA_VISIBLE_DEVICES=\"\""
        log_message "$YELLOW" "使用 CPU 模式"
    fi
    
    # 步骤1: 调优（tune）
    if [ "$SKIP_TUNE" = "true" ]; then
        log_message "$YELLOW" "[步骤 1/2] 跳过调优（使用已有的 best_params.json）"
    elif [ "$PARALLEL_TUNE" = "true" ]; then
        # 多GPU并行调优（仅限非Chemprop）
        log_message "$YELLOW" "\n[步骤 1/2] 开始并行调优: $current_config (Optuna, trials: $OPTUNA_TRIALS, 多GPU)"
        
        # 构建并行调优命令
        TUNE_CMD="$PYTHON_CMD -m src.parallel_tune --config $current_config --n-trials $OPTUNA_TRIALS --patience $PATIENCE --metric $METRIC"
        if [ -n "$N_GPUS" ]; then
            TUNE_CMD="$TUNE_CMD --n-gpus $N_GPUS"
        fi
        
        if eval "$cuda_env $TUNE_CMD" >> "$LOG_FILE" 2>&1; then
            log_message "$GREEN" "✓ 并行Optuna调优完成: $current_config"
        else
            log_message "$RED" "✗ 并行Optuna调优失败: $current_config"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            return 1
        fi
    else
        # 单GPU/CPU串行调优
        log_message "$YELLOW" "\n[步骤 1/2] 开始调优: $current_config (Optuna, trials: $OPTUNA_TRIALS)"
        TUNE_CMD="$PYTHON_CMD -m src.train --config $current_config --tune --optuna-trials $OPTUNA_TRIALS --patience $PATIENCE --metric $METRIC"
        
        if eval "$cuda_env $TUNE_CMD" >> "$LOG_FILE" 2>&1; then
            log_message "$GREEN" "✓ Optuna调优完成: $current_config"
        else
            log_message "$RED" "✗ Optuna调优失败: $current_config"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            return 1
        fi
    fi
    
    # 步骤2: 评估（evaluate）
    # 构建评估命令
    EVAL_CMD="$PYTHON_CMD -m src.train --config $current_config --n-seeds $N_SEEDS --patience $PATIENCE --metric $METRIC"
    if [ "$CLASS_WEIGHTS" = "true" ]; then
        EVAL_CMD="$EVAL_CMD --class-weights"
    fi
    if [ "$OPTIMIZE_THRESHOLD" = "true" ]; then
        EVAL_CMD="$EVAL_CMD --optimize-threshold"
    fi
    
    log_message "$YELLOW" "\n[步骤 2/2] 开始评估: $current_config (seeds: $N_SEEDS, class_weights: $CLASS_WEIGHTS, optimize_threshold: $OPTIMIZE_THRESHOLD)"
    if eval "$cuda_env $EVAL_CMD" >> "$LOG_FILE" 2>&1; then
        log_message "$GREEN" "✓ 评估完成: $current_config"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        return 0
    else
        log_message "$RED" "✗ 评估失败: $current_config"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

# 打印帮助信息
print_help() {
    echo "用法: [环境变量] ./batch_train.sh"
    echo ""
    echo "环境变量:"
    echo "  CONFIG              模型配置名称 (默认: GCN_DeepChem_Graph)"
    echo "  BATCH_MODE          批量模式: single | all_dgllife | all_chemprop | all (默认: single)"
    echo "  OPTUNA_TRIALS       Optuna超参数搜索次数 (默认: 50)"
    echo "  PATIENCE            早停耐心值 (默认: 10)"
    echo "  METRIC              早停指标: roc_auc | loss (默认: roc_auc)"
    echo "  N_SEEDS             随机种子数量 (默认: 10)"
    echo "  N_GPUS              GPU数量，空=使用全部"
    echo "  PARALLEL_TUNE       是否多GPU并行调优: true | false (默认: false)"
    echo "  CLASS_WEIGHTS       是否类别加权: true | false (默认: false)"
    echo "  OPTIMIZE_THRESHOLD  是否优化阈值: true | false (默认: false)"
    echo "  FORCE_CPU           强制使用CPU: true | false (默认: false)"
    echo "  SKIP_TUNE           跳过调优: true | false (默认: false)"
    echo ""
    echo "可用的模型配置:"
    echo "  DeepChem:  ${DEEPCHEM_CONFIGS[*]}"
    echo "  DGLlife:   ${DGLLIFE_CONFIGS[*]}"
    echo "  Chemprop:  ${CHEMPROP_CONFIGS[*]}"
    echo ""
    echo "示例:"
    echo "  ./batch_train.sh                                           # 默认配置"
    echo "  CONFIG=GAT_DGLlife_Graph ./batch_train.sh                  # 训练 GAT"
    echo "  CONFIG=DMPNN_Chemprop_Graph ./batch_train.sh               # 训练 D-MPNN"
    echo "  CONFIG=DMPNN_Chemprop_Graph+RDKit ./batch_train.sh         # D-MPNN + RDKit描述符融合"
    echo "  BATCH_MODE=all ./batch_train.sh                            # 训练所有模型"
    echo "  CLASS_WEIGHTS=true CONFIG=GAT_DGLlife_Graph ./batch_train.sh  # 类别加权"
}

# 主函数
main() {
    # 处理帮助参数
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        print_help
        exit 0
    fi
    
    # 确保在 dl 目录下运行
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR" || exit 1
    
    log_message "$GREEN" "=========================================="
    log_message "$GREEN" "深度学习模型训练脚本"
    log_message "$GREEN" "=========================================="
    log_message "$GREEN" "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log_message "$GREEN" "工作目录: $(pwd)"
    log_message "$GREEN" "配置: $CONFIG"
    log_message "$GREEN" "后端: $(get_backend "$CONFIG")"
    log_message "$GREEN" "融合特征: $(is_fusion_mode "$CONFIG")"
    log_message "$BLUE" "----------------------------------------"
    log_message "$GREEN" "Optuna试验次数: $OPTUNA_TRIALS"
    log_message "$GREEN" "早停耐心值: $PATIENCE"
    log_message "$GREEN" "早停指标: $METRIC"
    log_message "$GREEN" "并行调优: $PARALLEL_TUNE"
    log_message "$GREEN" "随机种子数量: $N_SEEDS"
    log_message "$GREEN" "类别加权损失: $CLASS_WEIGHTS"
    log_message "$GREEN" "阈值优化: $OPTIMIZE_THRESHOLD"
    log_message "$GREEN" "强制CPU模式: $FORCE_CPU"
    log_message "$GREEN" "跳过调优: $SKIP_TUNE"
    log_message "$GREEN" "批量模式: $BATCH_MODE"
    log_message "$GREEN" "日志文件: $LOG_FILE"
    
    # 检查 GPU 可用性
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log_message "$BLUE" "检测到 $GPU_COUNT 个 GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
            log_message "$BLUE" "  GPU: $line"
        done
        
        if [ "$PARALLEL_TUNE" = "true" ]; then
            if [ -n "$N_GPUS" ]; then
                log_message "$BLUE" "将使用 $N_GPUS 个 GPU 进行并行调优"
            else
                log_message "$BLUE" "将使用全部 $GPU_COUNT 个 GPU 进行并行调优"
            fi
        fi
    else
        log_message "$YELLOW" "警告: 未检测到 nvidia-smi，将使用 CPU"
        if [ "$PARALLEL_TUNE" = "true" ]; then
            log_message "$YELLOW" "回退到单进程调优"
            PARALLEL_TUNE="false"
        fi
    fi
    
    log_message "$GREEN" "=========================================="
    
    # 检查uv是否可用（仅用于创建虚拟环境）
    if ! command -v uv &> /dev/null; then
        log_message "$YELLOW" "警告: 未找到 'uv' 命令，将尝试使用现有的虚拟环境"
    fi
    
    # 检查输入目录
    if [ ! -d "input" ]; then
        log_message "$RED" "错误: 未找到 'input' 目录"
        exit 1
    fi
    
    # 根据批量模式运行训练
    case "$BATCH_MODE" in
        "all_dgllife")
            log_message "$BLUE" "\n批量训练模式: 将训练所有 DGLlife 模型"
            for cfg in "${DGLLIFE_CONFIGS[@]}"; do
                run_training "$cfg" || true
            done
            ;;
        "all_chemprop")
            log_message "$BLUE" "\n批量训练模式: 将训练所有 Chemprop 模型"
            for cfg in "${CHEMPROP_CONFIGS[@]}"; do
                run_training "$cfg" || true
            done
            ;;
        "all_deepchem")
            log_message "$BLUE" "\n批量训练模式: 将训练所有 DeepChem 模型"
            for cfg in "${DEEPCHEM_CONFIGS[@]}"; do
                run_training "$cfg" || true
            done
            ;;
        "all")
            log_message "$BLUE" "\n批量训练模式: 将训练所有模型"
            for cfg in "${ALL_CONFIGS[@]}"; do
                run_training "$cfg" || true
            done
            ;;
        *)
            # 单个模型训练
            run_training "$CONFIG" || true
            ;;
    esac
    
    # 计算总耗时
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # 打印最终统计
    log_message "$GREEN" "\n=========================================="
    log_message "$GREEN" "训练完成统计"
    log_message "$GREEN" "=========================================="
    log_message "$GREEN" "成功: $SUCCESS_COUNT"
    log_message "$RED" "失败: $FAIL_COUNT"
    if [ $SKIP_COUNT -gt 0 ]; then
        log_message "$YELLOW" "跳过: $SKIP_COUNT"
    fi
    log_message "$BLUE" "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    log_message "$BLUE" "日志文件: $LOG_FILE"
    log_message "$GREEN" "=========================================="
    
    # 如果有失败，返回非零退出码
    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# 运行主函数
main "$@"


#!/bin/bash
# ==============================================================================
# parallel_train_all.sh - 并行训练所有深度学习模型
# ==============================================================================
#
# 功能:
#   - 并行启动多个模型的训练任务
#   - 支持 DGLlife (GCN, GAT, Weave, AttentiveFP) 和 Chemprop (D-MPNN)
#   - 自动检测 GPU 并分配任务
#   - 支持特征融合模式 (D-MPNN + RDKit/ChemoPy2d)
#
# 使用方法:
#   ./parallel_train_all.sh                    # 训练所有模型
#   MODE=dgllife_only ./parallel_train_all.sh  # 仅 DGLlife 模型
#   MODE=chemprop_only ./parallel_train_all.sh # 仅 Chemprop 模型
#   MODE=fusion_only ./parallel_train_all.sh   # 仅融合特征模型
#
# ==============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 确保在 dl 目录下运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ==============================================================================
# 配置参数 - 默认值从 config.py 动态读取
# ==============================================================================

# 使用Python从config.py读取默认值的辅助函数
get_config_value() {
    local var_name=$1
    local default=$2
    .venv/bin/python -c "from src import config; print(getattr(config, '$var_name', $default))" 2>/dev/null || echo "$default"
}

# 检查.venv是否存在，如果存在则从config.py读取默认值
if [ -f ".venv/bin/python" ]; then
    DEFAULT_OPTUNA_TRIALS=$(get_config_value "OPTUNA_N_TRIALS" 100)
    DEFAULT_N_SEEDS=$(.venv/bin/python -c "from src import config; print(len(config.RANDOM_SEEDS))" 2>/dev/null || echo 10)
    DEFAULT_PATIENCE=$(.venv/bin/python -c "from src import config; print(config.TRAINING_PARAMS.get('early_stopping_patience', 25))" 2>/dev/null || echo 25)
else
    # 如果虚拟环境不存在，使用硬编码默认值（应与config.py保持一致）
    DEFAULT_OPTUNA_TRIALS=100
    DEFAULT_N_SEEDS=10
    DEFAULT_PATIENCE=25
fi

MODE="${MODE:-all}"                       # 模式: all | dgllife_only | chemprop_only | fusion_only
CLASS_WEIGHTS="${CLASS_WEIGHTS:-true}"    # 类别加权（推荐用于不平衡数据）
OPTIMIZE_THRESHOLD="${OPTIMIZE_THRESHOLD:-false}"  # 阈值优化
OPTUNA_TRIALS="${OPTUNA_TRIALS:-$DEFAULT_OPTUNA_TRIALS}"   # 从config.py读取
N_SEEDS="${N_SEEDS:-$DEFAULT_N_SEEDS}"                     # 从config.py读取
PATIENCE="${PATIENCE:-$DEFAULT_PATIENCE}"                  # 从config.py读取
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-true}"  # 是否等待所有任务完成

# 记录开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ==============================================================================
# 模型配置
# ==============================================================================

# DGLlife 模型（使用 GPU）
DGLLIFE_CONFIGS=(
    "GCN_DGLlife_Graph"
    "GAT_DGLlife_Graph"
    "Weave_DGLlife_Graph"
    "AttentiveFP_DGLlife_Graph"
)

# Chemprop 基础模型
CHEMPROP_BASE_CONFIGS=(
    "DMPNN_Chemprop_Graph"
)

# Chemprop 融合特征模型
CHEMPROP_FUSION_CONFIGS=(
    "DMPNN_Chemprop_Graph+RDKit"
    "DMPNN_Chemprop_Graph+ChemoPy2d"
)

# 根据模式选择配置
case "$MODE" in
    "dgllife_only")
        CONFIGS=("${DGLLIFE_CONFIGS[@]}")
        ;;
    "chemprop_only")
        CONFIGS=("${CHEMPROP_BASE_CONFIGS[@]}")
        ;;
    "fusion_only")
        CONFIGS=("${CHEMPROP_FUSION_CONFIGS[@]}")
        ;;
    "chemprop_all")
        CONFIGS=("${CHEMPROP_BASE_CONFIGS[@]}" "${CHEMPROP_FUSION_CONFIGS[@]}")
        ;;
    *)
        # all: 所有模型
        CONFIGS=(
            "${DGLLIFE_CONFIGS[@]}"
            "${CHEMPROP_BASE_CONFIGS[@]}"
            "${CHEMPROP_FUSION_CONFIGS[@]}"
        )
        ;;
esac

# 打印带颜色的消息
log_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 检测后端类型
get_backend() {
    local config=$1
    case $config in
        *_Chemprop_*)
            echo "chemprop"
            ;;
        *)
            echo "dgllife"
            ;;
    esac
}

# ==============================================================================
# 主流程
# ==============================================================================

log_message "$GREEN" "=========================================="
log_message "$GREEN" "并行训练所有模型"
log_message "$GREEN" "=========================================="
log_message "$GREEN" "时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_message "$GREEN" "工作目录: $(pwd)"
log_message "$GREEN" "模式: $MODE"
log_message "$GREEN" "时间戳: $TIMESTAMP"
log_message "$CYAN" "----------------------------------------"
log_message "$GREEN" "配置:"
log_message "$GREEN" "  CLASS_WEIGHTS: $CLASS_WEIGHTS"
log_message "$GREEN" "  OPTIMIZE_THRESHOLD: $OPTIMIZE_THRESHOLD"
log_message "$GREEN" "  OPTUNA_TRIALS: $OPTUNA_TRIALS"
log_message "$GREEN" "  N_SEEDS: $N_SEEDS"
log_message "$GREEN" "  PATIENCE: $PATIENCE"
log_message "$GREEN" "=========================================="

# 列出将要训练的模型
log_message "$BLUE" "\n将要训练的模型:"
for config in "${CONFIGS[@]}"; do
    backend=$(get_backend "$config")
    log_message "$CYAN" "  - $config ($backend)"
done

# 检查 GPU 可用性
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log_message "$BLUE" "\n检测到 $GPU_COUNT 个 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
        log_message "$BLUE" "  GPU: $line"
    done
else
    log_message "$YELLOW" "\n警告: 未检测到 nvidia-smi，将使用 CPU"
fi

log_message "$GREEN" "=========================================="

# 启动所有训练任务
PIDS=()
LOG_FILES=()
CONFIG_NAMES=()

log_message "$YELLOW" "\n启动训练任务..."
for config in "${CONFIGS[@]}"; do
    log_file="${config}_${TIMESTAMP}.log"
    backend=$(get_backend "$config")
    
    log_message "$BLUE" "  启动: $config ($backend)"
    log_message "$BLUE" "    日志: $log_file"
    
    # 构建环境变量
    env_vars="CONFIG=$config"
    env_vars="$env_vars CLASS_WEIGHTS=$CLASS_WEIGHTS"
    env_vars="$env_vars OPTIMIZE_THRESHOLD=$OPTIMIZE_THRESHOLD"
    env_vars="$env_vars OPTUNA_TRIALS=$OPTUNA_TRIALS"
    env_vars="$env_vars N_SEEDS=$N_SEEDS"
    env_vars="$env_vars PATIENCE=$PATIENCE"
    
    # Chemprop 现在支持 GPU，不再强制使用 CPU
    
    # 启动后台任务
    nohup bash -c "$env_vars ./batch_train.sh" > "$log_file" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    LOG_FILES+=("$log_file")
    CONFIG_NAMES+=("$config")
    log_message "$GREEN" "    PID: $PID"
done

# 保存 PIDs 到文件
PIDS_FILE="training_pids_${TIMESTAMP}.txt"
for i in "${!PIDS[@]}"; do
    echo "${PIDS[$i]} ${CONFIG_NAMES[$i]}" >> "$PIDS_FILE"
done
log_message "$BLUE" "\nPIDs 已保存到: $PIDS_FILE"

log_message "$GREEN" "\n=========================================="
log_message "$GREEN" "所有训练任务已启动"
log_message "$GREEN" "=========================================="

# 打印监控命令
log_message "$BLUE" "监控命令:"
log_message "$BLUE" "  查看进程: ps aux | grep batch_train.sh | grep -v grep"
log_message "$BLUE" "  查看日志: tail -f ${LOG_FILES[0]}"
log_message "$BLUE" "  查看所有日志: tail -f *_${TIMESTAMP}.log"
log_message "$BLUE" "  停止所有任务: pkill -f batch_train.sh"

# 如果需要等待完成
if [ "$WAIT_FOR_COMPLETION" = "true" ]; then
    log_message "$YELLOW" "\n等待所有任务完成..."
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        config=${CONFIG_NAMES[$i]}
        log_file=${LOG_FILES[$i]}
        
        log_message "$BLUE" "等待 $config (PID: $pid)..."
        if wait $pid; then
            log_message "$GREEN" "✓ $config 完成 (PID: $pid)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            exit_code=$?
            log_message "$RED" "✗ $config 失败 (PID: $pid, 退出码: $exit_code)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
    
    # 计算总耗时
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # 打印最终统计
    log_message "$GREEN" "\n=========================================="
    log_message "$GREEN" "所有任务完成"
    log_message "$GREEN" "=========================================="
    log_message "$GREEN" "成功: $SUCCESS_COUNT"
    log_message "$RED" "失败: $FAIL_COUNT"
    log_message "$BLUE" "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    log_message "$BLUE" "日志文件:"
    for log_file in "${LOG_FILES[@]}"; do
        log_message "$BLUE" "  - $log_file"
    done
    log_message "$GREEN" "=========================================="
    
    # 生成结果摘要
    log_message "$CYAN" "\n结果摘要:"
    for config in "${CONFIG_NAMES[@]}"; do
        if [ -f "output/$config/cv_summary.csv" ]; then
            auc=$(grep "external.*AUC" "output/$config/cv_summary.csv" 2>/dev/null | head -1)
            if [ -n "$auc" ]; then
                log_message "$GREEN" "  $config: $auc"
            else
                log_message "$YELLOW" "  $config: 训练完成，请查看 output/$config/cv_summary.csv"
            fi
        else
            log_message "$RED" "  $config: 未完成或失败"
        fi
    done
    
    # 发送通知（如果有配置）
    if [ -n "$PUSHOVER_TOKEN" ] && [ -n "$PUSHOVER_USER" ]; then
        MESSAGE="并行训练完成！(${MODE})
成功: $SUCCESS_COUNT
失败: $FAIL_COUNT
总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        
        curl -s \
            --form-string "token=$PUSHOVER_TOKEN" \
            --form-string "user=$PUSHOVER_USER" \
            --form-string "message=$MESSAGE" \
            --form-string "title=并行训练完成" \
            https://api.pushover.net/1/messages.json > /dev/null 2>&1
        
        log_message "$GREEN" "\n通知已发送"
    fi
    
    # 如果有失败，返回非零退出码
    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
else
    log_message "$GREEN" "\n任务已在后台运行，脚本退出。"
    log_message "$GREEN" "使用以上监控命令查看进度。"
    exit 0
fi

#!/bin/bash

# 批量训练脚本：7个模型 × 4个特征集 = 28种组合
# 每种组合先调优（tune）再评估（evaluate）
# 每个组合独立执行，失败不影响下一个

# 注意：不使用 set -e，以便每个组合的失败不影响下一个

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 模型列表（7个）- 可通过环境变量覆盖，逗号分隔
if [ -n "$MODELS" ]; then
    IFS=',' read -ra MODELS <<< "$MODELS"
else
    MODELS=("RF" "SVM" "XGB" "LGBM" "ADA" "QDA" "LDA")
fi

# 特征集列表（4个）- 可通过环境变量覆盖，逗号分隔
if [ -n "$DESCRIPTORS" ]; then
    IFS=',' read -ra DESCRIPTORS <<< "$DESCRIPTORS"
else
    DESCRIPTORS=("RDKit" "ChemoPy2d" "GraphOnly" "KlekotaRoth")
fi

# 配置 - 可通过环境变量覆盖
TUNING_METHOD="${TUNING_METHOD:-grid}"  # grid or optuna
OPTUNA_TRIALS="${OPTUNA_TRIALS:-50}"

# 创建日志文件（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="batch_training_${TIMESTAMP}.log"

# 统计变量
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#DESCRIPTORS[@]}))
CURRENT_COUNT=0
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

# 运行单个组合（调优 + 评估）
run_combination() {
    local model=$1
    local descriptor=$2
    local config="${model}_${descriptor}"
    CURRENT_COUNT=$((CURRENT_COUNT + 1))
    
    log_message "$BLUE" "\n=========================================="
    log_message "$BLUE" "[$CURRENT_COUNT/$TOTAL_COMBINATIONS] 开始处理: $config"
    log_message "$BLUE" "模型: $model | 特征集: $descriptor"
    log_message "$BLUE" "=========================================="
    
    # 检查是否已完成（可选：跳过已完成的组合）
    # 取消注释以下代码以启用跳过功能
    # if [ -f "output/$config/cv_summary.csv" ] && [ -f "output/$config/best_params.json" ]; then
    #     log_message "$YELLOW" "⚠  跳过 $config（已完成）"
    #     SKIP_COUNT=$((SKIP_COUNT + 1))
    #     return 0
    # fi
    
    # 步骤1: 调优（tune）
    log_message "$YELLOW" "\n[步骤 1/2] 开始调优: $config (方法: $TUNING_METHOD)"
    if [ "$TUNING_METHOD" = "optuna" ]; then
        if uv run python -m src.train --config "$config" --tune \
            --tuning-method optuna --optuna-trials "$OPTUNA_TRIALS" >> "$LOG_FILE" 2>&1; then
            log_message "$GREEN" "✓ Optuna调优完成: $config"
        else
            log_message "$RED" "✗ Optuna调优失败: $config"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            return 1
        fi
    else
        if uv run python -m src.train --config "$config" --tune >> "$LOG_FILE" 2>&1; then
            log_message "$GREEN" "✓ GridSearchCV调优完成: $config"
        else
            log_message "$RED" "✗ GridSearchCV调优失败: $config"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            return 1
        fi
    fi
    
    # 步骤2: 评估（evaluate）
    log_message "$YELLOW" "\n[步骤 2/2] 开始评估: $config"
    if uv run python -m src.train --config "$config" >> "$LOG_FILE" 2>&1; then
        log_message "$GREEN" "✓ 评估完成: $config"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        return 0
    else
        log_message "$RED" "✗ 评估失败: $config"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

# 主函数
main() {
    log_message "$GREEN" "=========================================="
    log_message "$GREEN" "批量训练脚本启动"
    log_message "$GREEN" "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log_message "$GREEN" "总组合数: $TOTAL_COMBINATIONS"
    log_message "$GREEN" "调优方法: $TUNING_METHOD"
    if [ "$TUNING_METHOD" = "optuna" ]; then
        log_message "$GREEN" "Optuna试验次数: $OPTUNA_TRIALS"
    fi
    log_message "$GREEN" "日志文件: $LOG_FILE"
    log_message "$GREEN" "=========================================="
    
    # 检查uv是否可用
    if ! command -v uv &> /dev/null; then
        log_message "$RED" "错误: 未找到 'uv' 命令"
        log_message "$RED" "请确保 uv 已安装并在 PATH 中"
        exit 1
    fi
    
    # 检查输入目录
    if [ ! -d "input" ]; then
        log_message "$RED" "错误: 未找到 'input' 目录"
        exit 1
    fi
    
    # 遍历所有组合
    for model in "${MODELS[@]}"; do
        for descriptor in "${DESCRIPTORS[@]}"; do
            # 使用子shell运行，确保错误不会终止整个脚本
            if run_combination "$model" "$descriptor"; then
                : # 成功，继续
            else
                : # 失败已记录，继续下一个
            fi
        done
    done
    
    # 计算总耗时
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # 打印最终统计
    log_message "$GREEN" "\n=========================================="
    log_message "$GREEN" "批量训练完成"
    log_message "$GREEN" "=========================================="
    log_message "$BLUE" "总组合数: $TOTAL_COMBINATIONS"
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
main

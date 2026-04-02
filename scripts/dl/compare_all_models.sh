#!/bin/bash
# 对比所有模型的性能

echo "=========================================="
echo "模型性能对比 (External Set)"
echo "=========================================="
printf "%-30s %10s %10s %10s\n" "模型" "AUC (%)" "ACC (%)" "F1 (%)"
echo "----------------------------------------------------------"

CONFIGS=(
    "GCN_DGLlife_Graph"
    "GAT_DGLlife_Graph"
    "Weave_DGLlife_Graph"
    "AttentiveFP_DGLlife_Graph"
    "DMPNN_Chemprop_Graph"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "output/$config/cv_summary.csv" ]; then
        auc=$(grep "external" output/$config/cv_summary.csv | grep "AUC" | cut -d',' -f2)
        acc=$(grep "external" output/$config/cv_summary.csv | grep "ACC" | cut -d',' -f2)
        f1=$(grep "external" output/$config/cv_summary.csv | grep "F1" | cut -d',' -f2)
        printf "%-30s %10.2f %10.2f %10.2f\n" "$config" "$auc" "$acc" "$f1"
    else
        printf "%-30s %10s %10s %10s\n" "$config" "N/A" "N/A" "N/A"
    fi
done

echo "=========================================="
echo ""
echo "详细结果查看:"
echo "  output/<模型名>/cv_summary.csv"

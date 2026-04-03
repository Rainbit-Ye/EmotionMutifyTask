#!/bin/bash
# 数据预处理运行脚本

echo "========================================"
echo "DailyDialog 数据预处理"
echo "========================================"

cd /home/user1/liuduanye/EmotionClassify

# 运行数据处理脚本
python process_database/process_dailydialog.py

echo ""
echo "数据处理完成！"
echo "数据文件保存在: /home/user1/liuduanye/EmotionClassify/data/"

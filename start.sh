#!/bin/bash
# FastAPI 学习服务一键启动脚本
# 用法：Git Bash 里在任意目录执行  bash /c/code/Langchain/start.sh
cd "$(dirname "$0")"                # 回到仓库根目录，从哪里启动都行
source venv/Scripts/activate        # 激活项目自己的虚拟环境（提示符应显示 (venv)）
unset SSL_CERT_FILE                 # conda base 激活时会把它指向一个不存在的文件，会让 ChatOpenAI 构造崩溃
cd FastAPI
echo ">>> 启动中：服务 http://127.0.0.1:8000 | 文档 http://127.0.0.1:8000/docs"
uvicorn main:app --reload

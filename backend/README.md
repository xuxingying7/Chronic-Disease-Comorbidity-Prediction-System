# 机器学习后端（Flask）

## 运行

- Windows（推荐 PowerShell）
  1. 进入 `backend` 目录
  2. 创建虚拟环境并安装依赖
     ```powershell
     python -m venv venv
     ./venv/Scripts/pip.exe install -r requirements.txt
     ```
  3. 启动
     ```powershell
     ./venv/Scripts/python.exe app.py
     ```

- 或者直接运行根目录的 `start.ps1`，会自动在后台启动后端。

## API

- GET `/api/health` 健康检查
- POST `/api/upload` 上传Excel
- POST `/api/preprocess` 预处理（多重插补+划分）
- POST `/api/train_models` 训练模型（XGBoost/随机森林/朴素贝叶斯/逻辑回归/决策树/MLP）

## 数据要求

- Excel 首行为列名，最后一列为标签。


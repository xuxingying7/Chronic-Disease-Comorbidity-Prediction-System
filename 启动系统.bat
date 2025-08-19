@echo off
chcp 65001 >nul
echo ========================================
echo        疾病预测系统完整启动器
echo ========================================
echo.
echo 本脚本将启动完整的疾病预测系统
echo 包括：后端API服务 + 前端Web应用
echo.

set "CURRENT_DIR=%cd%"

echo [1/4] 检查系统环境...
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    echo ✓ Python已安装
)

:: 检查Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到Node.js，请先安装Node.js
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
) else (
    echo ✓ Node.js已安装
)

echo.
echo [2/4] 设置后端环境...
echo.

:: 进入backend目录并设置环境
cd /d "%CURRENT_DIR%\backend"
if not exist "venv" (
    echo 正在创建Python虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
)

echo ✓ 检测到依赖包已安装，跳过安装步骤
echo ✓ 后端环境设置完成

echo.
echo [3/4] 设置前端环境...
echo.

:: 进入web目录并安装依赖
cd /d "%CURRENT_DIR%\web"
if not exist "node_modules" (
    echo 正在安装前端依赖...
    npm install
    if errorlevel 1 (
        echo ❌ 前端依赖安装失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 前端依赖已安装
)

echo.
echo [4/4] 启动系统服务...
echo.

:: 启动后端API（在新窗口中）
echo 启动后端API服务器...
cd /d "%CURRENT_DIR%\backend"
start "疾病预测系统-后端API" cmd /k "echo 后端API服务器 & echo API地址: http://localhost:5000 & echo 健康检查: http://localhost:5000/api/health & echo. & python app.py"

:: 等待后端启动
echo 等待后端服务启动...
timeout /t 5 /nobreak > nul

:: 启动前端应用
echo 启动前端Web应用...
cd /d "%CURRENT_DIR%\web"
start "疾病预测系统-前端应用" cmd /k "echo 前端Web应用 & echo 应用地址: http://localhost:3000 & echo. & npm start"

echo.
echo ========================================
echo         系统启动完成！
echo ========================================
echo.
echo 🚀 后端API服务: http://localhost:5000
echo 🌐 前端Web应用: http://localhost:3000
echo.
echo 📝 使用说明:
echo 1. 等待前端应用在浏览器中自动打开
echo 2. 点击"数据分析"菜单开始使用
echo 3. 可以上传Excel文件进行机器学习分析
echo.
echo 💡 提示:
echo - 如需停止服务，关闭对应的命令行窗口即可
echo - 详细使用说明请查看"机器学习集成说明.md"
echo - 依赖验证说明请查看"依赖验证说明.md"
echo.

pause

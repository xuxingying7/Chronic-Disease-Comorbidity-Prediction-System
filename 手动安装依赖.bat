@echo off
chcp 65001 >nul
echo ========================================
echo        Python依赖手动安装工具
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ 检测到Python
python --version
echo.

:: 进入backend目录
cd /d "%~dp0backend"
if not exist "requirements.txt" (
    echo ❌ 未找到requirements.txt文件
    pause
    exit /b 1
)

echo 当前目录: %cd%
echo.

:: 选择安装方式
echo 请选择安装方式:
echo 1. 检查已安装的依赖包
echo 2. 创建虚拟环境并使用官方源安装
echo 3. 创建虚拟环境并使用清华镜像安装
echo 4. 创建虚拟环境并使用阿里云镜像安装
echo 5. 直接安装到系统Python环境（不推荐）
echo 6. 退出
echo.
set /p choice="请输入选择 (1-6): "

if "%choice%"=="1" goto :check_installed
if "%choice%"=="2" goto :install_official
if "%choice%"=="3" goto :install_tsinghua
if "%choice%"=="4" goto :install_aliyun
if "%choice%"=="5" goto :install_system
if "%choice%"=="6" goto :exit
goto :invalid_choice

:check_installed
echo.
echo 正在检查已安装的依赖包...
echo.
pip list | findstr /i "flask flask-cors pandas numpy scikit-learn openpyxl xgboost"
echo.
echo 如果上述包都已显示，说明依赖已安装完成
echo 可以直接运行 "启动系统.bat" 启动系统
echo.
pause
goto :exit

:install_official
echo.
echo 正在使用官方源安装...
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
goto :check_result

:install_tsinghua
echo.
echo 正在使用清华镜像安装...
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
goto :check_result

:install_aliyun
echo.
echo 正在使用阿里云镜像安装...
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
goto :check_result

:install_system
echo.
echo 正在安装到系统Python环境...
pip install --upgrade pip
pip install -r requirements.txt
goto :check_result

:check_result
if errorlevel 1 (
    echo.
    echo ❌ 安装失败！
    echo.
    echo 可能的解决方案:
    echo 1. 检查网络连接
    echo 2. 尝试使用不同的镜像源
    echo 3. 手动安装单个包:
    echo    pip install flask flask-cors pandas numpy scikit-learn openpyxl
    echo 4. 如果xgboost安装失败，可以跳过: pip install flask flask-cors pandas numpy scikit-learn openpyxl
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ✓ 安装成功！
    echo.
    echo 现在可以启动系统了:
    echo 1. 运行 "启动系统.bat" 启动完整系统
    echo 2. 或运行 "start.bat" 启动前端
    echo.
)

:exit
echo 按任意键退出...
pause >nul
exit /b 0

:invalid_choice
echo 无效选择，请重新运行脚本
pause
exit /b 1

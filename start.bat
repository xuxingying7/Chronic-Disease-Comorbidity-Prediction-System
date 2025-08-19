@echo off
chcp 65001 >nul
echo ========================================
echo        疾病预测系统启动器
echo ========================================
echo.

:: 检查Node.js是否安装
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Node.js，请先安装Node.js
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

echo [信息] Node.js版本: 
node --version
echo.

:: 检测可用的包管理器
set "PACKAGE_MANAGER="

:: 检查pnpm
pnpm --version >nul 2>&1
if not errorlevel 1 (
    set "PACKAGE_MANAGER=pnpm"
    echo [信息] 检测到pnpm，将使用pnpm作为包管理器
    goto :install_deps
)

:: 检查yarn
yarn --version >nul 2>&1
if not errorlevel 1 (
    set "PACKAGE_MANAGER=yarn"
    echo [信息] 检测到yarn，将使用yarn作为包管理器
    goto :install_deps
)

:: 检查npm
npm --version >nul 2>&1
if not errorlevel 1 (
    set "PACKAGE_MANAGER=npm"
    echo [信息] 检测到npm，将使用npm作为包管理器
    goto :install_deps
)

:: 如果没有检测到任何包管理器
echo [错误] 未检测到任何包管理器（npm/yarn/pnpm）
echo 请安装以下任一包管理器：
echo - npm (随Node.js一起安装)
echo - yarn: npm install -g yarn
echo - pnpm: npm install -g pnpm
pause
exit /b 1

:install_deps
echo.
echo [信息] 正在安装项目依赖...
echo.

if "%PACKAGE_MANAGER%"=="pnpm" (
    pnpm install
) else if "%PACKAGE_MANAGER%"=="yarn" (
    yarn install
) else (
    npm install
)

if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请检查网络连接或手动安装
    pause
    exit /b 1
)

echo.
echo [成功] 依赖安装完成！
echo.

:: 启动Python后端（如存在）
if exist backend\app.py (
    echo [信息] 检测到Python后端，准备启动API服务...
    pushd backend
    echo [信息] 检测到依赖包已安装，直接启动API服务...
    start /min python app.py
    popd
)

echo [信息] 正在启动开发服务器...
echo [信息] 应用将在浏览器中自动打开
echo [信息] 按 Ctrl+C 可以停止服务器
echo.

if "%PACKAGE_MANAGER%"=="pnpm" (
    pnpm start
) else if "%PACKAGE_MANAGER%"=="yarn" (
    yarn start
) else (
    npm start
)

pause


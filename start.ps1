# 疾病预测系统启动器 (PowerShell版本)
# 支持 npm, yarn, pnpm

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "        疾病预测系统启动器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 若存在Python后端，则在后台启动
if (Test-Path -Path "backend/app.py") {
    Write-Host "[信息] 检测到Python后端，准备启动API服务..." -ForegroundColor Yellow
    Push-Location backend
    Write-Host "[信息] 检测到依赖包已安装，直接启动API服务..." -ForegroundColor Green
    Start-Process -FilePath "python" -ArgumentList "app.py" -WindowStyle Minimized
    Pop-Location
}

# 检查Node.js是否安装
try {
    $nodeVersion = node --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[信息] Node.js版本: $nodeVersion" -ForegroundColor Green
    } else {
        throw "Node.js not found"
    }
} catch {
    Write-Host "[错误] 未检测到Node.js，请先安装Node.js" -ForegroundColor Red
    Write-Host "下载地址: https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

Write-Host ""

# 检测可用的包管理器
$packageManager = $null

# 检查pnpm
try {
    $pnpmVersion = pnpm --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $packageManager = "pnpm"
        Write-Host "[信息] 检测到pnpm，将使用pnpm作为包管理器" -ForegroundColor Green
    }
} catch {}

# 检查yarn
if (-not $packageManager) {
    try {
        $yarnVersion = yarn --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $packageManager = "yarn"
            Write-Host "[信息] 检测到yarn，将使用yarn作为包管理器" -ForegroundColor Green
        }
    } catch {}
}

# 检查npm
if (-not $packageManager) {
    try {
        $npmVersion = npm --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $packageManager = "npm"
            Write-Host "[信息] 检测到npm，将使用npm作为包管理器" -ForegroundColor Green
        }
    } catch {}
}

# 如果没有检测到任何包管理器
if (-not $packageManager) {
    Write-Host "[错误] 未检测到任何包管理器（npm/yarn/pnpm）" -ForegroundColor Red
    Write-Host "请安装以下任一包管理器：" -ForegroundColor Yellow
    Write-Host "- npm (随Node.js一起安装)" -ForegroundColor White
    Write-Host "- yarn: npm install -g yarn" -ForegroundColor White
    Write-Host "- pnpm: npm install -g pnpm" -ForegroundColor White
    Read-Host "按回车键退出"
    exit 1
}

Write-Host ""
Write-Host "[信息] 正在安装项目依赖..." -ForegroundColor Yellow
Write-Host ""

# 安装依赖
try {
    switch ($packageManager) {
        "pnpm" { pnpm install }
        "yarn" { yarn install }
        "npm" { npm install }
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed"
    }
    
    Write-Host ""
    Write-Host "[成功] 依赖安装完成！" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "[信息] 正在启动开发服务器..." -ForegroundColor Yellow
    Write-Host "[信息] 应用将在浏览器中自动打开" -ForegroundColor Cyan
    Write-Host "[信息] 按 Ctrl+C 可以停止服务器" -ForegroundColor Cyan
    Write-Host ""
    
    # 启动开发服务器
    switch ($packageManager) {
        "pnpm" { pnpm start }
        "yarn" { yarn start }
        "npm" { npm start }
    }
    
} catch {
    Write-Host ""
    Write-Host "[错误] 操作失败: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请检查网络连接或手动安装依赖" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

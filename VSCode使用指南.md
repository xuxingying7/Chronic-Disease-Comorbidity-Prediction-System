# VSCode 使用指南 - 前端小白专用

## 🚀 快速开始

### 第一步：打开项目
1. 打开VSCode
2. 按 `Ctrl+K Ctrl+O` 或者点击 `File` → `Open Folder`
3. 选择项目中的 `web` 文件夹
4. 点击 `Select Folder`

### 第二步：打开终端
- 按 `Ctrl+`` `（反引号键，通常在数字1的左边）
- 或者点击顶部菜单 `Terminal` → `New Terminal`

### 第三步：安装依赖
在终端中输入以下命令（选择其中一个）：

```bash
# 推荐使用pnpm（如果已安装）
pnpm install

# 或者使用yarn
yarn install

# 或者使用npm
npm install
```

**等待安装完成**（可能需要几分钟，取决于网络速度）

### 第四步：启动项目
安装完成后，在终端中输入：

```bash
# 使用pnpm
pnpm start

# 或者使用yarn
yarn start

# 或者使用npm
npm start
```

### 第五步：查看结果
- 浏览器会自动打开，显示地址：http://localhost:3000
- 如果没有自动打开，手动在浏览器中输入上述地址

## 🔧 常用VSCode快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+`` ` | 打开/关闭终端 |
| `Ctrl+S` | 保存文件 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Shift+P` | 打开命令面板 |
| `F5` | 调试 |
| `Ctrl+F` | 查找 |
| `Ctrl+H` | 替换 |

## 📁 项目文件结构说明

```
web/
├── src/                    # 源代码文件夹
│   ├── components/         # 组件文件夹
│   │   ├── Header.js       # 头部组件
│   │   ├── DataUpload.js   # 数据上传组件
│   │   ├── DataAnalysis.js # 数据分析组件
│   │   └── Prediction.js   # 预测组件
│   ├── App.js             # 主应用文件
│   ├── App.css            # 主样式文件
│   ├── index.js           # 入口文件
│   └── index.css          # 全局样式
├── public/                # 静态资源
│   └── index.html         # HTML模板
├── package.json           # 项目配置文件
├── start.bat             # 批处理启动脚本
├── start.ps1             # PowerShell启动脚本
└── README.md             # 项目说明
```

## 🛠️ 开发技巧

### 1. 实时预览
- 启动后，修改代码会自动刷新浏览器
- 无需手动刷新页面

### 2. 查看控制台
- 在浏览器中按 `F12` 打开开发者工具
- 查看 `Console` 标签页的错误信息

### 3. 代码提示
- VSCode会自动提供代码补全
- 输入时会显示相关函数和属性

### 4. 文件导航
- 按 `Ctrl+P` 快速搜索文件
- 按 `Ctrl+Shift+F` 全局搜索

## ❗ 常见问题

### 问题1：终端显示"命令未找到"
**解决方案：**
- 确保已安装Node.js
- 在终端中输入 `node --version` 检查

### 问题2：安装依赖失败
**解决方案：**
- 检查网络连接
- 尝试使用国内镜像：
  ```bash
  npm config set registry https://registry.npmmirror.com
  ```

### 问题3：端口被占用
**解决方案：**
- 关闭其他可能占用3000端口的程序
- 或者修改端口号

### 问题4：浏览器没有自动打开
**解决方案：**
- 手动在浏览器中输入：http://localhost:3000

## 🎯 下一步学习建议

1. **学习React基础**：了解组件、状态、属性等概念
2. **学习JavaScript ES6**：箭头函数、解构赋值、模块导入等
3. **学习CSS**：样式布局和响应式设计
4. **学习调试技巧**：使用浏览器开发者工具

## 📞 获取帮助

如果遇到问题：
1. 查看浏览器控制台的错误信息
2. 检查终端输出的错误信息
3. 搜索相关错误信息
4. 查看项目文档

祝您学习愉快！🎉

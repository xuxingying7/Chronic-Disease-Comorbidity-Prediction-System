# 疾病预测系统

基于机器学习的疾病预测系统，专门用于分析老年人群中多病共存现象的影响因素。

## 项目背景

老年人群中多病共存现象日益普遍，给公共卫生体系带来了重大挑战。尽管先前的研究已强调了个体行为的作用，但生活方式因素与社会经济地位在多病共存中的复杂交互作用仍不清楚。

本项目基于"中国健康和退休纵向研究"(CHARLS)的具有全国代表性的数据，构建了机器学习模型来预测慢性病风险。

## 功能特性

### 1. 数据上传
- 支持Excel格式(.xlsx, .xls)文件上传
- 自动数据验证和格式检查
- 数据预览和统计信息展示

### 2. 数据分析
- 数据可视化展示
- 变量相关性分析
- 统计信息汇总

### 3. 疾病预测
- 基于机器学习模型的预测
- 个性化风险评估
- 预测结果可视化

## 数据要求

上传的Excel文件应包含以下变量：

### 自变量
- Gender (性别)
- Age (年龄)
- Vision problem (视力问题)
- Disability (残疾)
- Self-assessment of health status (自评健康状况)
- Depression (抑郁)
- Physical exercise (体育锻炼)
- Nap time (午睡时间)
- Sleep time (睡眠时间)
- Social activities (社交活动)
- Daily activity ability (日常活动能力)
- Marital status (婚姻状况)
- Distribution (分布)
- Residence (居住地)
- Educational status (教育状况)
- Work (工作)
- Per capita annual income (Yuan) (人均年收入)

### 因变量
- Chronic (慢性病)

## 技术栈

- **前端框架**: React 18
- **UI组件库**: Ant Design 5
- **图表库**: ECharts
- **文件处理**: XLSX
- **路由**: React Router DOM
- **HTTP客户端**: Axios

## 安装和运行

### 环境要求
- Node.js 16.0 或更高版本
- npm 或 yarn

### 安装依赖
```bash
npm install
```

### 启动开发服务器
```bash
npm start
```

应用将在 http://localhost:3000 启动

### 构建生产版本
```bash
npm run build
```

## 项目结构

```
web/
├── public/                 # 静态资源
├── src/
│   ├── components/         # React组件
│   │   ├── Header.js      # 页面头部
│   │   ├── DataUpload.js  # 数据上传
│   │   ├── DataAnalysis.js # 数据分析
│   │   └── Prediction.js  # 疾病预测
│   ├── App.js             # 主应用组件
│   ├── App.css            # 应用样式
│   ├── index.js           # 应用入口
│   └── index.css          # 全局样式
├── package.json           # 项目配置
└── README.md             # 项目说明
```

## 使用说明

1. **数据上传**: 在首页上传符合格式要求的Excel文件
2. **数据分析**: 查看数据统计信息和可视化图表
3. **疾病预测**: 输入个人信息进行慢性病风险预测

## 注意事项

- 确保上传的Excel文件包含所有必要的变量列
- 数据格式应保持一致，避免缺失值过多
- 预测结果仅供参考，不能替代专业医疗诊断

## 开发团队

本项目基于CHARLS研究数据开发，旨在为公共卫生研究提供技术支持。

## 许可证

MIT License


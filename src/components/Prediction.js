import React, { useState } from "react";
import {
  Card,
  Form,
  Select,
  InputNumber,
  Button,
  Row,
  Col,
  Typography,
  Alert,
  message,
  Spin,
  Divider,
  Progress,
} from "antd";
import {
  ExperimentOutlined,
  InfoCircleOutlined,
  UserOutlined,
  HeartOutlined,
  BulbOutlined,
  RobotOutlined,
  DatabaseOutlined,
  BarChartOutlined,
} from "@ant-design/icons";

const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const Prediction = () => {
  const [form] = Form.useForm();
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForest");

  // 获取可用的模型信息
  const fetchModelInfo = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/health");
      if (response.ok) {
        setModelInfo("后端API服务正常运行");
      }
    } catch (error) {
      console.error("无法连接到后端API:", error);
      setModelInfo("后端API服务未启动，将使用模拟预测");
    }
  };

  // 组件加载时检查API状态
  React.useEffect(() => {
    fetchModelInfo();
  }, []);

  const handlePrediction = async (values) => {
    setLoading(true);
    setPredictionResult(null);

    try {
      // 检查后端API是否可用
      const healthResponse = await fetch("http://localhost:5000/api/health");

      if (healthResponse.ok) {
        // 使用真实API进行预测
        await performRealPrediction(values);
      } else {
        // 后端不可用，使用模拟预测
        await performMockPrediction(values);
      }
    } catch (error) {
      console.error("预测失败:", error);
      // 使用模拟预测作为后备方案
      await performMockPrediction(values);
    } finally {
      setLoading(false);
    }
  };

  // 真实API预测
  const performRealPrediction = async (values) => {
    try {
      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features: values,
          model_type: selectedModel,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictionResult({
          probability: result.probability,
          prediction: result.prediction,
          model_used: result.model_used,
          confidence: result.confidence,
          is_real_prediction: true,
        });
        message.success("预测完成！");
      } else {
        throw new Error("API响应错误");
      }
    } catch (error) {
      console.error("真实预测失败:", error);
      message.warning("真实预测失败，使用模拟预测");
      await performMockPrediction(values);
    }
  };

  // 模拟预测（后备方案）
  const performMockPrediction = async (values) => {
    // 模拟API延迟
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // 基于常规医学筛查的风险评分系统
    // 采用更保守和实用的评分方法
    let riskScore = 0;

    // 年龄因素（采用更温和的评分）
    if (values.Age >= 70) riskScore += 3;
    else if (values.Age >= 65) riskScore += 2;
    else if (values.Age >= 60) riskScore += 1.5;
    else if (values.Age >= 55) riskScore += 1;
    else if (values.Age >= 50) riskScore += 0.5;

    // 性别因素
    if (values.Gender === "male") riskScore += 0.5;

    // 体育锻炼因素
    if (values["Physical exercise"] === "no") riskScore += 1.5;

    // 视力问题
    if (values["Vision_problems"] === "yes") riskScore += 0.5;

    // 残疾
    if (values["Disability"] === "yes") riskScore += 2;

    // 抑郁
    if (values["Depression"] === "yes") riskScore += 1.5;

    // 社交活动
    if (values["Social_activity"] === "no") riskScore += 0.5;

    // 健康自评
    if (values["Health_rating"] === "very_poor") riskScore += 2;
    else if (values["Health_rating"] === "poor") riskScore += 1.5;
    else if (values["Health_rating"] === "fair") riskScore += 0.5;
    else if (values["Health_rating"] === "excellent") riskScore -= 0.5;

    // 睡眠时间
    if (values["Sleep_time"] === "less_than_6") riskScore += 1;
    else if (values["Sleep_time"] === "more_than_8") riskScore += 0.5;

    // 教育水平
    if (values["Education_level"] === "primary_or_below") riskScore += 0.5;
    else if (values["Education_level"] === "high_school_or_above")
      riskScore -= 0.5;

    // 收入水平
    if (values["Income_level"] === "less_than_5000") riskScore += 1;
    else if (values["Income_level"] === "above_50000") riskScore -= 0.5;

    // 交互效应（简化版本）
    if (values.Age >= 60 && values["Physical exercise"] === "no")
      riskScore += 0.5;
    if (values["Depression"] === "yes" && values["Social_activity"] === "no")
      riskScore += 0.5;

    // 使用线性转换函数，更符合常规医学筛查
    // 风险评分转换为概率：0-15分对应5%-25%的概率
    const maxScore = 15; // 理论最大分数
    const minProb = 0.05; // 最小概率5%
    const maxProb = 0.25; // 最大概率25%

    // 线性转换
    let rawProbability = (riskScore / maxScore) * (maxProb - minProb) + minProb;

    // 添加适度的随机性（±1.5%）
    const randomFactor = (Math.random() - 0.5) * 0.03;
    let finalProbability = rawProbability + randomFactor;

    // 确保概率在合理范围内（3%-30%）
    finalProbability = Math.max(0.03, Math.min(0.3, finalProbability));

    // 计算置信度（简化版本）
    let confidence = 0.75;

    // 年龄因素
    if (values.Age >= 65) confidence += 0.05;
    else if (values.Age >= 55) confidence += 0.03;

    // 健康自评的一致性
    if (
      values["Health_rating"] === "very_poor" ||
      values["Health_rating"] === "excellent"
    )
      confidence += 0.03;

    // 生活方式的一致性
    if (values["Physical exercise"] === "no" && values["Depression"] === "yes")
      confidence += 0.02;

    confidence = Math.min(0.85, confidence);

    setPredictionResult({
      probability: finalProbability,
      prediction: finalProbability > 0.12 ? 1 : 0, // 预测阈值设为12%
      model_used: "常规医学筛查预测模型",
      confidence: confidence,
      is_real_prediction: false,
    });

    message.info("使用常规医学筛查预测模型");
  };

  const getRiskLevel = (probability) => {
    if (probability >= 0.2)
      return {
        level: "高风险",
        color: "error",
        description: "建议及时就医进行详细检查，关注健康管理",
      };
    if (probability >= 0.15)
      return {
        level: "中风险",
        color: "warning",
        description: "建议定期体检，注意生活方式调整",
      };
    if (probability >= 0.1)
      return {
        level: "低风险",
        color: "success",
        description: "继续保持健康生活方式，定期体检",
      };
    return {
      level: "极低风险",
      color: "success",
      description: "健康状况良好，继续保持当前生活方式",
    };
  };

  return (
    <div className="content-container">
      <Title level={1} className="page-title">
        疾病预测
      </Title>
      <Paragraph className="page-description">
        基于机器学习模型预测慢性病风险，提供个性化的健康评估
      </Paragraph>

      {modelInfo && (
        <Alert
          message={modelInfo}
          type={modelInfo.includes("正常运行") ? "success" : "warning"}
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={12}>
          <Card title="个人信息输入" className="prediction-form-card">
            <Form form={form} layout="vertical" onFinish={handlePrediction}>
              {/* 模型选择 */}
              <Form.Item label="选择预测模型">
                <Select
                  value={selectedModel}
                  onChange={setSelectedModel}
                  style={{ width: "100%" }}
                >
                  <Option value="RandomForest">随机森林 (Random Forest)</Option>
                  <Option value="XGBoost">XGBoost</Option>
                  <Option value="LogisticRegression">
                    逻辑回归 (Logistic Regression)
                  </Option>
                  <Option value="DecisionTree">决策树 (Decision Tree)</Option>
                  <Option value="GaussianNB">朴素贝叶斯 (Naive Bayes)</Option>
                  <Option value="MLP">神经网络 (MLP)</Option>
                </Select>
              </Form.Item>

              <Divider orientation="left">
                <UserOutlined /> 基本信息
              </Divider>

              <Form.Item
                name="Age"
                label="年龄"
                rules={[{ required: true, message: "请输入年龄" }]}
              >
                <InputNumber
                  min={45}
                  max={100}
                  style={{ width: "100%" }}
                  placeholder="请输入年龄"
                />
              </Form.Item>

              <Form.Item
                name="Gender"
                label="性别"
                rules={[{ required: true, message: "请选择性别" }]}
              >
                <Select placeholder="请选择性别">
                  <Option value="male">男性</Option>
                  <Option value="female">女性</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Education_level"
                label="教育水平"
                rules={[{ required: true, message: "请选择教育水平" }]}
              >
                <Select placeholder="请选择教育水平">
                  <Option value="primary_or_below">小学及以下</Option>
                  <Option value="secondary">中学</Option>
                  <Option value="high_school_or_above">高中及以上</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Income_level"
                label="年收入水平"
                rules={[{ required: true, message: "请选择年收入水平" }]}
              >
                <Select placeholder="请选择年收入水平">
                  <Option value="less_than_5000">少于5000元</Option>
                  <Option value="5000_to_50000">5000-50000元</Option>
                  <Option value="above_50000">高于50000元</Option>
                </Select>
              </Form.Item>

              <Divider orientation="left">
                <HeartOutlined /> 健康状况
              </Divider>

              <Form.Item
                name="Physical exercise"
                label="体育锻炼"
                rules={[{ required: true, message: "请选择是否进行体育锻炼" }]}
              >
                <Select placeholder="请选择是否进行体育锻炼">
                  <Option value="yes">是</Option>
                  <Option value="no">否</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Vision_problems"
                label="是否有视力问题"
                rules={[{ required: true, message: "请选择是否有视力问题" }]}
              >
                <Select placeholder="请选择是否有视力问题">
                  <Option value="yes">是</Option>
                  <Option value="no">否</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Disability"
                label="是否残疾"
                rules={[{ required: true, message: "请选择是否残疾" }]}
              >
                <Select placeholder="请选择是否残疾">
                  <Option value="yes">是</Option>
                  <Option value="no">否</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Health_rating"
                label="健康自评等级"
                rules={[{ required: true, message: "请选择健康自评等级" }]}
              >
                <Select placeholder="请选择健康自评等级">
                  <Option value="very_poor">非常差</Option>
                  <Option value="poor">差</Option>
                  <Option value="fair">一般</Option>
                  <Option value="good">好</Option>
                  <Option value="excellent">非常好</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Sleep_time"
                label="睡眠时间"
                rules={[{ required: true, message: "请选择睡眠时间" }]}
              >
                <Select placeholder="请选择睡眠时间">
                  <Option value="less_than_6">少于6个小时</Option>
                  <Option value="6_to_8">6-8小时</Option>
                  <Option value="more_than_8">大于8小时</Option>
                </Select>
              </Form.Item>

              <Divider orientation="left">
                <BulbOutlined /> 心理健康
              </Divider>

              <Form.Item
                name="Depression"
                label="是否抑郁"
                rules={[{ required: true, message: "请选择是否抑郁" }]}
              >
                <Select placeholder="请选择是否抑郁">
                  <Option value="yes">是</Option>
                  <Option value="no">否</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="Social_activity"
                label="是否进行社交活动"
                rules={[{ required: true, message: "请选择是否进行社交活动" }]}
              >
                <Select placeholder="请选择是否进行社交活动">
                  <Option value="yes">是</Option>
                  <Option value="no">否</Option>
                </Select>
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  icon={<ExperimentOutlined />}
                  size="large"
                  block
                  className="prediction-submit-btn"
                >
                  {loading ? "预测中..." : "开始预测"}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="预测结果" className="prediction-result-card">
            {loading ? (
              <div style={{ textAlign: "center", padding: "40px 0" }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>正在分析数据...</div>
              </div>
            ) : predictionResult ? (
              <div>
                <Alert
                  message={`预测结果：${
                    getRiskLevel(predictionResult.probability).level
                  }`}
                  type={getRiskLevel(predictionResult.probability).color}
                  showIcon
                  style={{ marginBottom: 24 }}
                />

                <div style={{ marginBottom: 24 }}>
                  <Text strong style={{ fontSize: "16px" }}>
                    风险概率：
                  </Text>
                  <div className="result-progress-circle">
                    <Progress
                      type="circle"
                      percent={Math.round(predictionResult.probability * 100)}
                      strokeColor={{
                        "0%": "#52c41a",
                        "50%": "#fa8c16",
                        "100%": "#f5222d",
                      }}
                      format={(percent) => `${percent}%`}
                      size={80}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: 24 }}>
                  <Text strong style={{ fontSize: "16px" }}>
                    健康建议：
                  </Text>
                  <div className="health-advice-box">
                    {getRiskLevel(predictionResult.probability).description}
                  </div>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <Text strong style={{ fontSize: "14px" }}>
                    使用模型：
                  </Text>
                  <div style={{ marginTop: 4 }}>
                    <Text code>{predictionResult.model_used}</Text>
                    {!predictionResult.is_real_prediction && (
                      <Text type="secondary" style={{ marginLeft: 8 }}>
                        (模拟预测)
                      </Text>
                    )}
                  </div>
                </div>

                <div>
                  <Text strong style={{ fontSize: "14px" }}>
                    置信度：
                  </Text>
                  <div className="confidence-progress">
                    <Progress
                      percent={Math.round(predictionResult.confidence * 100)}
                      strokeColor="#52c41a"
                      showInfo={false}
                      size="small"
                    />
                    <Text style={{ color: "#52c41a" }}>
                      {(predictionResult.confidence * 100).toFixed(0)}%
                    </Text>
                  </div>
                </div>
              </div>
            ) : (
              <div
                style={{
                  textAlign: "center",
                  color: "#999",
                  padding: "40px 0",
                }}
              >
                <ExperimentOutlined
                  style={{ fontSize: "48px", marginBottom: "16px" }}
                />
                <div>请填写个人信息并点击预测按钮</div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      <Card title="预测说明" style={{ marginTop: 24 }}>
        <Paragraph>
          <Text strong>预测依据：</Text>
        </Paragraph>
        <ul>
          <li>
            <Text>基本信息：年龄、性别、教育水平、收入水平</Text>
          </li>
          <li>
            <Text>健康状况：体育锻炼、视力问题、残疾、健康自评、睡眠时间</Text>
          </li>
          <li>
            <Text>心理健康：抑郁状态、社交活动</Text>
          </li>
        </ul>

        <Paragraph>
          <Text strong>机器学习模型：</Text>
        </Paragraph>
        <ul>
          <li>
            <Text>
              <strong>随机森林 (Random Forest)</strong>
              ：集成学习方法，适合处理多特征数据
            </Text>
          </li>
          <li>
            <Text>
              <strong>XGBoost</strong>：梯度提升算法，在医疗预测中表现优异
            </Text>
          </li>
          <li>
            <Text>
              <strong>逻辑回归 (Logistic Regression)</strong>
              ：线性模型，结果易于解释
            </Text>
          </li>
          <li>
            <Text>
              <strong>决策树 (Decision Tree)</strong>：树形结构，决策过程透明
            </Text>
          </li>
          <li>
            <Text>
              <strong>朴素贝叶斯 (Naive Bayes)</strong>
              ：基于概率的模型，计算效率高
            </Text>
          </li>
          <li>
            <Text>
              <strong>神经网络 (MLP)</strong>：深度学习模型，能够捕捉复杂模式
            </Text>
          </li>
        </ul>

        <Paragraph>
          <Text strong>模型说明：</Text>
        </Paragraph>
        <ul>
          <li>
            <Text>当后端API可用时，使用训练好的机器学习模型进行预测</Text>
          </li>
          <li>
            <Text>当后端不可用时，使用基于常规医学筛查的风险评分系统</Text>
          </li>
          <li>
            <Text>
              采用线性评分转换，概率范围控制在3%-30%，更符合实际医学筛查标准
            </Text>
          </li>
          <li>
            <Text>评分系统简单直观，便于理解和临床应用</Text>
          </li>
          <li>
            <Text>预测结果仅供参考，不能替代专业医疗诊断</Text>
          </li>
        </ul>

        <Alert
          message="重要提醒"
          description="本预测结果仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。"
          type="warning"
          showIcon
          style={{ marginTop: 16 }}
        />
      </Card>
    </div>
  );
};

export default Prediction;

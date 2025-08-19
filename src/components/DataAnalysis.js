import React, { useState, useEffect } from "react";
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Table,
  Button,
  Upload,
  message,
  Spin,
  Alert,
  Tabs,
  Progress,
} from "antd";
import {
  InboxOutlined,
  ExperimentOutlined,
  RobotOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";
import axios from "axios";

const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Dragger } = Upload;

// API基础URL
const API_BASE_URL = "http://localhost:5000/api";

const DataAnalysis = () => {
  // 状态管理
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [preprocessedData, setPreprocessedData] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [activeTab, setActiveTab] = useState("basic");
  const pieData = [
    { name: "慢性病患者", value: 870, color: "#1890ff" },
    { name: "非慢性病患者", value: 780, color: "#40a9ff" },
  ];

  const correlationData = [
    { variable: "年龄", correlation: 0.65, pValue: 0.001 },
    { variable: "性别", correlation: 0.32, pValue: 0.05 },
    { variable: "视力问题", correlation: 0.58, pValue: 0.001 },
    { variable: "残疾", correlation: 0.72, pValue: 0.001 },
    { variable: "自评健康状况", correlation: 0.68, pValue: 0.001 },
    { variable: "抑郁", correlation: 0.61, pValue: 0.001 },
    { variable: "体育锻炼", correlation: -0.45, pValue: 0.01 },
    { variable: "社交活动", correlation: -0.38, pValue: 0.05 },
    { variable: "教育状况", correlation: -0.42, pValue: 0.01 },
    { variable: "收入水平", correlation: -0.35, pValue: 0.05 },
  ];

  const correlationColumns = [
    {
      title: "变量",
      dataIndex: "variable",
      key: "variable",
      align: "center",
    },
    {
      title: "相关系数",
      dataIndex: "correlation",
      key: "correlation",
      align: "center",
      render: (value) => value.toFixed(3),
    },
    {
      title: "P值",
      dataIndex: "pValue",
      key: "pValue",
      align: "center",
      render: (value) => value.toFixed(3),
    },
  ];

  const getPieChartOption = () => ({
    title: {
      text: "慢性病患病率分布",
      left: "center",
      textStyle: {
        fontSize: 20,
        fontWeight: "bold",
        color: "#1890ff",
      },
    },
    tooltip: {
      trigger: "item",
      formatter: function (params) {
        return `${params.seriesName}<br/>${params.name}: ${params.value} (${params.percent}%)`;
      },
    },
    legend: {
      orient: "vertical",
      left: "left",
      bottom: "bottom",
      textStyle: {
        fontSize: 16,
        color: "#666",
      },
    },
    series: [
      {
        name: "患病情况",
        type: "pie",
        radius: ["40%", "70%"],
        center: ["50%", "45%"],
        data: pieData,
        itemStyle: {
          borderRadius: 8,
          borderColor: "#fff",
          borderWidth: 2,
        },
        label: {
          show: true,
          formatter: "{b}: {c} ({d}%)",
          fontSize: 14,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: "rgba(0, 0, 0, 0.5)",
          },
        },
      },
    ],
  });

  // 文件上传处理
  const handleFileUpload = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setUploadedFile(response.data);
      message.success("文件上传成功！");
      setActiveTab("ml");
    } catch (error) {
      message.error(
        "文件上传失败: " + (error.response?.data?.error || error.message)
      );
    } finally {
      setLoading(false);
    }

    return false; // 阻止默认上传行为
  };

  // 数据预处理
  const handlePreprocess = async () => {
    if (!uploadedFile) {
      message.error("请先上传数据文件");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/preprocess`, {
        filepath: uploadedFile.data_info.filepath,
      });

      setPreprocessedData(response.data);
      message.success("数据预处理完成！");
    } catch (error) {
      message.error(
        "数据预处理失败: " + (error.response?.data?.error || error.message)
      );
    } finally {
      setLoading(false);
    }
  };

  // 模型训练
  const handleModelTraining = async () => {
    if (!preprocessedData) {
      message.error("请先进行数据预处理");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/train_models`, {
        train_path: preprocessedData.train_path,
        test_path: preprocessedData.test_path,
      });

      setModelResults(response.data.results);
      message.success("模型训练完成！");
    } catch (error) {
      message.error(
        "模型训练失败: " + (error.response?.data?.error || error.message)
      );
    } finally {
      setLoading(false);
    }
  };

  // 获取模型性能图表配置
  const getModelComparisonChart = () => {
    if (!modelResults) return {};

    const models = Object.keys(modelResults).filter(
      (key) => !modelResults[key].error
    );
    const accuracyData = models.map(
      (model) => modelResults[model].test_metrics?.accuracy || 0
    );
    const f1Data = models.map(
      (model) => modelResults[model].test_metrics?.f1_score || 0
    );

    return {
      title: {
        text: "模型性能比较",
        left: "center",
        textStyle: {
          fontSize: 20,
          fontWeight: "bold",
          color: "#1890ff",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: function (params) {
          let result = params[0].name + "<br/>";
          params.forEach((param) => {
            result +=
              param.seriesName +
              ": " +
              (param.value * 100).toFixed(2) +
              "%<br/>";
          });
          return result;
        },
      },
      legend: {
        data: ["准确率", "F1分数"],
        top: 40,
        textStyle: {
          fontSize: 16,
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "15%",
        top: "20%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: models,
        axisLabel: {
          rotate: 45,
          fontSize: 14,
          interval: 0,
          textStyle: {
            color: "#666",
          },
        },
        axisTick: {
          alignWithLabel: true,
        },
      },
      yAxis: {
        type: "value",
        max: 1,
        axisLabel: {
          formatter: function (value) {
            return (value * 100).toFixed(0) + "%";
          },
          fontSize: 14,
        },
      },
      series: [
        {
          name: "准确率",
          type: "bar",
          data: accuracyData,
          itemStyle: {
            color: "#1890ff",
          },
          barWidth: "40%",
        },
        {
          name: "F1分数",
          type: "bar",
          data: f1Data,
          itemStyle: {
            color: "#faad14",
          },
          barWidth: "40%",
        },
      ],
    };
  };

  // 渲染模型结果表格
  const renderModelResultsTable = () => {
    if (!modelResults) return null;

    const tableData = Object.entries(modelResults).map(
      ([modelName, result]) => {
        if (result.error) {
          return {
            key: modelName,
            model: modelName,
            status: "失败",
            error: result.error,
            accuracy: "-",
            precision: "-",
            recall: "-",
            f1_score: "-",
          };
        }

        const testMetrics = result.test_metrics || {};
        return {
          key: modelName,
          model: modelName,
          status: "成功",
          accuracy: (testMetrics.accuracy * 100).toFixed(2) + "%",
          precision: (testMetrics.precision * 100).toFixed(2) + "%",
          recall: (testMetrics.recall * 100).toFixed(2) + "%",
          f1_score: (testMetrics.f1_score * 100).toFixed(2) + "%",
        };
      }
    );

    const columns = [
      { title: "模型", dataIndex: "model", key: "model", align: "center" },
      { title: "状态", dataIndex: "status", key: "status", align: "center" },
      {
        title: "准确率",
        dataIndex: "accuracy",
        key: "accuracy",
        align: "center",
      },
      {
        title: "精确率",
        dataIndex: "precision",
        key: "precision",
        align: "center",
      },
      { title: "召回率", dataIndex: "recall", key: "recall", align: "center" },
      {
        title: "F1分数",
        dataIndex: "f1_score",
        key: "f1_score",
        align: "center",
      },
    ];

    return (
      <Table
        columns={columns}
        dataSource={tableData}
        pagination={false}
        size="small"
      />
    );
  };

  return (
    <div className="content-container">
      <Title level={1} className="page-title">
        智能数据分析
      </Title>
      <Paragraph className="page-description">
        基于机器学习的慢性病共病预测分析
      </Paragraph>

      <Spin spinning={loading} tip="处理中...">
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* 基础分析标签页 */}
          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                基础分析
              </span>
            }
            key="basic"
          >
            <Row gutter={[24, 24]}>
              <Col xs={12} lg={6}>
                <Card className="analysis-stats-card">
                  <Statistic title="总样本数" value={1650} />
                </Card>
              </Col>
              <Col xs={12} lg={6}>
                <Card className="analysis-stats-card">
                  <Statistic title="慢性病患者" value={870} />
                </Card>
              </Col>
              <Col xs={12} lg={6}>
                <Card className="analysis-stats-card">
                  <Statistic title="非慢性病患者" value={780} />
                </Card>
              </Col>
              <Col xs={12} lg={6}>
                <Card className="analysis-stats-card">
                  <Statistic title="患病率" value={52.7} suffix="%" />
                </Card>
              </Col>
            </Row>

            <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
              <Col xs={24} lg={12}>
                <Card
                  title="慢性病患病率分布"
                  className="chart-container"
                  style={{
                    height: "500px",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <ReactECharts
                      option={getPieChartOption()}
                      style={{ height: "400px", width: "100%" }}
                    />
                  </div>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card
                  title="变量与慢性病的相关性分析"
                  style={{
                    height: "500px",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <Table
                      columns={correlationColumns}
                      dataSource={correlationData}
                      pagination={false}
                      size="middle"
                      scroll={{ y: 300 }}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* 机器学习分析标签页 */}
          <TabPane
            tab={
              <span>
                <RobotOutlined />
                机器学习分析
              </span>
            }
            key="ml"
          >
            <Row gutter={[24, 24]}>
              {/* 数据上传区域 */}
              <Col xs={24} lg={8}>
                <Card
                  title="数据上传"
                  extra={<InboxOutlined />}
                  style={{
                    height: "400px",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "space-between",
                    }}
                  >
                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                      }}
                    >
                      <Dragger
                        name="file"
                        multiple={false}
                        accept=".xlsx,.xls"
                        beforeUpload={handleFileUpload}
                        showUploadList={false}
                        style={{
                          flex: 1,
                          display: "flex",
                          flexDirection: "column",
                          justifyContent: "center",
                        }}
                      >
                        <p className="ant-upload-drag-icon">
                          <InboxOutlined />
                        </p>
                        <p className="ant-upload-text">
                          点击或拖拽Excel文件到此区域上传
                        </p>
                        <p className="ant-upload-hint">
                          支持 .xlsx 和 .xls 格式
                        </p>
                      </Dragger>
                    </div>

                    {uploadedFile && (
                      <Alert
                        style={{ marginTop: 16, marginBottom: 0 }}
                        message={`文件已上传: ${uploadedFile.data_info.filename}`}
                        description={`数据形状: ${uploadedFile.data_info.shape[0]}行 × ${uploadedFile.data_info.shape[1]}列`}
                        type="success"
                        showIcon
                      />
                    )}
                  </div>
                </Card>
              </Col>

              {/* 数据预处理区域 */}
              <Col xs={24} lg={8}>
                <Card
                  title="数据预处理"
                  extra={<ExperimentOutlined />}
                  style={{
                    height: "400px",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "space-between",
                    }}
                  >
                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <Button
                        type="primary"
                        icon={<ExperimentOutlined />}
                        onClick={handlePreprocess}
                        disabled={!uploadedFile}
                        style={{ marginBottom: 16 }}
                      >
                        开始预处理
                      </Button>
                      <div
                        style={{
                          color: "#666",
                          fontSize: "12px",
                          marginBottom: 16,
                        }}
                      >
                        包括缺失值插补和数据划分
                      </div>
                    </div>

                    {preprocessedData && (
                      <Alert
                        style={{ marginTop: 16, marginBottom: 0 }}
                        message="预处理完成"
                        description={
                          <div>
                            <div>
                              训练集: {preprocessedData.train_shape[0]}行
                            </div>
                            <div>
                              测试集: {preprocessedData.test_shape[0]}行
                            </div>
                            <div>
                              缺失值: {preprocessedData.missing_values_before} →{" "}
                              {preprocessedData.missing_values_after}
                            </div>
                          </div>
                        }
                        type="success"
                        showIcon
                      />
                    )}
                  </div>
                </Card>
              </Col>

              {/* 模型训练区域 */}
              <Col xs={24} lg={8}>
                <Card
                  title="模型训练"
                  extra={<RobotOutlined />}
                  style={{
                    height: "400px",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "space-between",
                    }}
                  >
                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <Button
                        type="primary"
                        icon={<RobotOutlined />}
                        onClick={handleModelTraining}
                        disabled={!preprocessedData}
                        style={{ marginBottom: 16 }}
                      >
                        训练模型
                      </Button>
                      <div
                        style={{
                          color: "#666",
                          fontSize: "12px",
                          marginBottom: 16,
                        }}
                      >
                        训练6种机器学习模型
                      </div>
                    </div>

                    {modelResults && (
                      <Alert
                        style={{ marginTop: 16, marginBottom: 0 }}
                        message="模型训练完成"
                        description={`成功训练${
                          Object.keys(modelResults).filter(
                            (k) => !modelResults[k].error
                          ).length
                        }个模型`}
                        type="success"
                        showIcon
                      />
                    )}
                  </div>
                </Card>
              </Col>
            </Row>

            {/* 模型结果展示 */}
            {modelResults && (
              <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
                <Col xs={24} lg={12}>
                  <Card
                    title="模型性能比较"
                    style={{
                      height: "500px",
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <ReactECharts
                        option={getModelComparisonChart()}
                        style={{ height: "400px", width: "100%" }}
                      />
                    </div>
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card
                    title="详细评估结果"
                    style={{
                      height: "500px",
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <div style={{ flex: 1 }}>{renderModelResultsTable()}</div>
                  </Card>
                </Col>
              </Row>
            )}
          </TabPane>
        </Tabs>
      </Spin>
    </div>
  );
};

export default DataAnalysis;

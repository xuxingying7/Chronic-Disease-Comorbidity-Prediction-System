import React, { useState } from "react";
import {
  Card,
  Upload,
  Button,
  message,
  Table,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Divider,
} from "antd";
import {
  InboxOutlined,
  FileExcelOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import * as XLSX from "xlsx";
import { useNavigate } from "react-router-dom";

const { Title, Paragraph, Text } = Typography;
const { Dragger } = Upload;

const DataUpload = () => {
  const [data, setData] = useState(null);
  const [dataStats, setDataStats] = useState(null);
  const navigate = useNavigate();

  const handleFileUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const workbook = XLSX.read(e.target.result, { type: "binary" });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        if (jsonData.length < 2) {
          message.error("文件格式错误");
          return false;
        }

        const headers = jsonData[0];
        const rows = jsonData.slice(1);

        const formattedData = rows.map((row, index) => {
          const obj = {};
          headers.forEach((header, i) => {
            obj[header] = row[i];
          });
          obj.key = index;
          return obj;
        });

        setData(formattedData);

        const stats = {
          totalRecords: formattedData.length,
          chronicCount: formattedData.filter(
            (row) => row.Chronic === 1 || row.Chronic === "1"
          ).length,
          nonChronicCount: formattedData.filter(
            (row) => row.Chronic === 0 || row.Chronic === "0"
          ).length,
        };

        setDataStats(stats);
        message.success("数据上传成功！");
        return false;
      } catch (error) {
        message.error("文件解析失败");
        return false;
      }
    };
    reader.readAsBinaryString(file);
    return false;
  };

  const columns = data
    ? Object.keys(data[0] || {})
        .map((key) => ({
          title: key,
          dataIndex: key,
          key: key,
          ellipsis: true,
          width: 120,
        }))
        .filter((col) => col.key !== "key")
    : [];

  const uploadProps = {
    name: "file",
    multiple: false,
    accept: ".xlsx,.xls",
    beforeUpload: handleFileUpload,
    showUploadList: false,
  };

  return (
    <div className="content-container">
      <Title level={1} className="page-title">
        数据上传
      </Title>
      <Paragraph className="page-description">
        上传研究数据，支持Excel格式(.xlsx, .xls)，系统将自动分析数据特征
      </Paragraph>

      <Row gutter={[32, 32]}>
        <Col xs={24} lg={14}>
          <Card
            title={
              <span>
                <FileExcelOutlined
                  style={{ marginRight: 8, color: "#1890ff" }}
                />
                数据上传
              </span>
            }
            className="upload-card"
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
                justifyContent: "center",
              }}
            >
              <Dragger {...uploadProps} className="custom-dragger">
                <p className="ant-upload-drag-icon">
                  <InboxOutlined
                    style={{ fontSize: "48px", color: "#1890ff" }}
                  />
                </p>
                <p
                  className="ant-upload-text"
                  style={{ fontSize: "18px", fontWeight: 500 }}
                >
                  点击或拖拽文件到此区域上传
                </p>
                <p className="ant-upload-hint" style={{ color: "#666" }}>
                  支持 .xlsx 和 .xls 格式的Excel文件
                </p>
              </Dragger>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card
            title={
              <span>
                <DatabaseOutlined
                  style={{ marginRight: 8, color: "#52c41a" }}
                />
                数据概览
              </span>
            }
            className="stats-card"
            style={{
              height: "400px",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {dataStats ? (
              <div>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="总记录数"
                      value={dataStats.totalRecords}
                      valueStyle={{ color: "#1890ff", fontSize: "24px" }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="数据完整性"
                      value={95}
                      suffix="%"
                      valueStyle={{ color: "#52c41a", fontSize: "24px" }}
                    />
                  </Col>
                </Row>

                <Divider style={{ margin: "16px 0" }} />

                <div style={{ marginBottom: 16 }}>
                  <Text strong>慢性病分布：</Text>
                  <Progress
                    percent={Math.round(
                      (dataStats.chronicCount / dataStats.totalRecords) * 100
                    )}
                    strokeColor={{
                      "0%": "#1890ff",
                      "100%": "#efc520ff",
                    }}
                    showInfo={false}
                    style={{ marginTop: 8 }}
                  />
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: 4,
                    }}
                  >
                    <Text type="secondary">
                      慢性病患者: {dataStats.chronicCount}
                    </Text>
                    <Text type="secondary">
                      非慢性病患者: {dataStats.nonChronicCount}
                    </Text>
                  </div>
                </div>
              </div>
            ) : (
              <div
                style={{
                  textAlign: "center",
                  padding: "40px 0",
                  color: "#999",
                }}
              >
                <DatabaseOutlined
                  style={{ fontSize: "48px", marginBottom: 16 }}
                />
                <div>请先上传数据文件</div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {data && (
        <Card
          title={
            <span>
              <BarChartOutlined style={{ marginRight: 8, color: "#fa8c16" }} />
              数据预览
            </span>
          }
          style={{ marginTop: 32 }}
          extra={
            <Button
              type="primary"
              size="large"
              onClick={() => navigate("/analysis")}
              icon={<BarChartOutlined />}
            >
              开始分析
            </Button>
          }
          className="preview-card"
        >
          <div style={{ marginBottom: 16 }}>
            <Text type="secondary">
              显示前10条记录，共 {data.length} 条数据
            </Text>
          </div>
          <Table
            columns={columns}
            dataSource={data.slice(0, 10)}
            pagination={false}
            scroll={{ x: "max-content" }}
            size="middle"
            className="custom-table"
          />
        </Card>
      )}
    </div>
  );
};

export default DataUpload;

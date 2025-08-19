import React from "react";
import { Layout, Menu } from "antd";
import { useNavigate, useLocation } from "react-router-dom";
import {
  UploadOutlined,
  BarChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";

const { Header } = Layout;

const HeaderComponent = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: "/",
      icon: <UploadOutlined />,
      label: "数据上传",
    },
    {
      key: "/analysis",
      icon: <BarChartOutlined />,
      label: "数据分析",
    },
    {
      key: "/prediction",
      icon: <ExperimentOutlined />,
      label: "疾病预测",
    },
  ];

  const handleMenuClick = ({ key }) => {
    navigate(key);
  };

  return (
    <Header>
      <div className="logo">基于机器学习的慢性病共病预测系统</div>
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        className="nav-menu"
      />
    </Header>
  );
};

export default HeaderComponent;

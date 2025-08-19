import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import Header from './components/Header';
import DataUpload from './components/DataUpload';
import DataAnalysis from './components/DataAnalysis';
import Prediction from './components/Prediction';
import './App.css';

const { Content } = Layout;

function App() {
  return (
    <Router>
      <Layout className="medical-layout">
        <div className="background-animation">
          <div className="floating-shapes">
            <div className="shape shape-1"></div>
            <div className="shape shape-2"></div>
            <div className="shape shape-3"></div>
            <div className="shape shape-4"></div>
            <div className="shape shape-5"></div>
          </div>
        </div>
        <Header />
        <Content className="main-content">
          <Routes>
            <Route path="/" element={<DataUpload />} />
            <Route path="/analysis" element={<DataAnalysis />} />
            <Route path="/prediction" element={<Prediction />} />
          </Routes>
        </Content>
      </Layout>
    </Router>
  );
}

export default App;

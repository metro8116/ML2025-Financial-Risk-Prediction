# Financial Risk Prediction | 金融风控违约预测

## 项目简介 (Project Overview)

本项目是**机器学习项目**。

**项目背景：**
金融风险评估是金融科技领域的关键问题。本项目基于某金融机构提供的用户贷款数据，利用机器学习算法预测用户是否存在违约风险（`isDefault`），并输出预测概率及特征重要性分析，为资产安全和收益控制提供技术支持。

**主要成果：**
- 构建了基于 **XGBoost** 的二分类预测模型。
- 实现了完整的数据预处理流水线（缺失值填补、Target Encoding、标准化）。
- 在 5 折交叉验证中取得了 **0.7351** 的平均 AUC 分数。

## 实验环境与依赖 (Requirements)

本项目基于 Windows 11 + Anaconda 开发，主要依赖库版本如下：

- **Python**: 3.12
- **XGBoost**: 3.0.2
- **Pandas**: 2.2.2
- **Scikit-learn**: 1.5.1
- **Category Encoders**: 2.8.1
- **Matplotlib**: 3.9.2

安装依赖：
```bash
pip install pandas numpy xgboost scikit-learn category_encoders matplotlib

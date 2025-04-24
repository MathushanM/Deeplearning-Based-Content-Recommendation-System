# 📚 Deep Learning-Based Content Recommendation System for E-Learning

This project presents a personalized content recommendation engine for an e-learning platform using deep learning techniques. It implements and compares two prominent models — **Neural Collaborative Filtering (NCF)** and **Light Graph Convolutional Network (LightGCN)** — to recommend educational content to users based on their engagement patterns.

---

## 🚀 Project Overview

The system is designed to address challenges of sparsity and personalization in online learning environments. Key features include:

- Comparative implementation of **NCF** and **LightGCN**
- Performance evaluation using **accuracy, precision, recall, NDCG, MAP, and ROC-AUC**
- Integration-ready for use with a web-based e-learning platform
- Visualizations of model results using bar plots and heatmaps

---

## 📁 Dataset

The dataset used is the [**Online Course Engagement Dataset**](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset) from Kaggle, which includes features such as course completion, views, user feedback, and engagement scores.

---

## 🧠 Models Implemented

### ✅ Neural Collaborative Filtering (NCF)
- Implemented in TensorFlow/Keras
- Embedding-based user-item interaction
- Dense MLP layers for non-linear learning

### ✅ LightGCN
- Implemented in PyTorch
- Simplified graph convolution architecture
- Optimized for sparse recommendation tasks

---

## 📊 Evaluation Metrics

| Metric       | Description                                  |
|--------------|----------------------------------------------|
| Accuracy     | Classification accuracy for implicit ratings |
| Precision@K  | Relevance of top-K recommendations           |
| Recall@K     | Coverage of relevant items in top-K          |
| NDCG@K       | Ranking quality of recommendations           |
| MAP@K        | Mean Average Precision                       |
| ROC-AUC      | Binary classification performance (NCF)      |

---

## 📈 Results Summary

| Model     | Precision | Recall | NDCG  | MAP   | ROC-AUC | Accuracy |
|-----------|-----------|--------|-------|--------|----------|----------|
| NCF       | 0.5723    | 0.5403 | —     | —      | 0.6226   | 0.5653   |
| LightGCN  | 0.6800    | 0.6800 | 3.628 | 0.6521 | —        | —        |

LightGCN outperformed NCF in sparse and top-K ranking scenarios, making it a better fit for educational recommendation use cases.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (for evaluation metrics)

---

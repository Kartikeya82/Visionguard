🛡️ Intelligent CCTV Surveillance System Using Deep Learning

An AI-powered smart surveillance system for real-time human activity recognition and anomaly detection from CCTV video streams.
This project evaluates and deploys state-of-the-art hybrid deep learning architectures to balance accuracy, temporal intelligence, and real-world performance for smart-city security applications.

📌 Project Overview

Traditional CCTV surveillance relies heavily on manual monitoring, leading to delayed responses, human fatigue, and inconsistent threat detection.
This project addresses these limitations by introducing an automated, intelligent video surveillance pipeline capable of understanding spatio-temporal human behavior using deep learning.

The system processes CCTV footage as 16-frame video sequences, extracts spatial features, models temporal motion, and classifies activities in real time.

🎯 Key Objectives

Automate human activity recognition from CCTV footage

Detect abnormal or suspicious activities using temporal modeling

Compare hybrid deep learning architectures for real-world deployment

Optimize inference speed without sacrificing accuracy

Design a scalable and deployment-ready surveillance pipeline

🔍 System Capabilities

📹 Video-based Human Activity Recognition

🧠 Spatio-Temporal Feature Learning

🚨 Anomaly & Suspicious Activity Detection

⚡ Real-Time Optimized Inference

📊 Comprehensive Performance Evaluation

🌐 Smart-City Ready Surveillance Framework

🧠 Models Implemented

The project evaluates three hybrid deep learning architectures:

1️⃣ Vision Transformer + ConvLSTM (ViT-ConvLSTM)

Strong global spatial reasoning

Limited generalization on small CCTV datasets

High computational cost

Training Accuracy: 82.20%
Inference Speed: Slow

2️⃣ ConvNeXt + LSTM ⭐ (Best Model)

CNN efficiency with transformer-level performance

Strong temporal modeling and fast inference

Best balance between accuracy and speed

Training Accuracy: 93.16%
Inference Speed: Fastest ⚡
Best for Real-Time Deployment

3️⃣ EfficientNet3D + ConvLSTM

Powerful spatio-temporal feature extraction

Severe overfitting and high computation cost

Not suitable for real-time use

Training Accuracy: 95%
Generalization: Poor
Inference Speed: Very Slow 🐌

🏗️ System Pipeline
CCTV Video
   ↓
Frame Extraction (16-frame sequences)
   ↓
Preprocessing (Resize, CLAHE, Normalization)
   ↓
Spatial Feature Extraction (CNN / Transformer)
   ↓
Temporal Modeling (LSTM / ConvLSTM)
   ↓
Activity Classification (Softmax)
   ↓
Output: Activity / Anomaly Label

📊 Dataset Details

Custom CCTV action recognition dataset

Sequence length: 16 frames per clip

Frame size: 224 × 224

Data Split:

80% Training

10% Validation

10% Testing

Preprocessing Techniques:

CLAHE (contrast enhancement)

Brightness normalization

Frame stabilization

Tensor normalization

📈 Evaluation Metrics

Accuracy

RMSE

MAE

MAPE

Confusion Matrix

Inference Speed

🏆 Key Results
Model	Accuracy	Inference Speed	Generalization
ViT + ConvLSTM	82.20%	Slow	Medium
ConvNeXt + LSTM	93.16%	Fastest	Best
EfficientNet3D + ConvLSTM	95% (Train)	Very Slow	Poor

✅ ConvNeXt-LSTM emerged as the most stable, accurate, and deployment-ready model.

🚀 Tech Stack

Programming Language:

Python

Deep Learning & CV:

TensorFlow / Keras

OpenCV

Data Handling & Visualization:

NumPy

Pandas

Matplotlib

Model Types:

CNN

Vision Transformer (ViT)

LSTM

ConvLSTM

3D CNN

🧪 Research Contributions

Designed and evaluated three hybrid deep learning architectures

Built an optimized end-to-end CCTV activity recognition pipeline

Analyzed accuracy vs inference speed trade-offs

Identified ConvNeXt-LSTM as the best real-world solution

Provided insights for smart-city surveillance deployment

🔮 Future Enhancements

Integration of Temporal Transformers

Larger and more diverse CCTV datasets

Advanced anomaly scoring mechanisms

Real-time alerting dashboards

Edge-device deployment optimization

👥 Authors

Kartikeya Singh
Tanishq Kakkar

📄 License

This project is intended for academic, research, and educational purposes.
For commercial usage, please contact the authors.

# 🫁 Pneumonia Detection from Chest X-rays

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/) [![Streamlit](https://img.shields.io/badge/streamlit-%E2%9C%85-green.svg)](https://streamlit.io/)

---

## 🚀 Project Overview

This project builds a **deep learning model** to accurately detect **Pneumonia** from **Chest X-ray images** using **MobileNetV2** with transfer learning. A user-friendly **Streamlit** app lets you upload images and get instant predictions!

---

## 📂 Project Structure

├── App.py                # Streamlit app for inference  
├── model/                # Model folder  
│   └── best_model.h5     # Trained MobileNetV2 model  
├── notebooks/            # Jupyter notebooks for training & evaluation  
├── requirements.txt      # Required packages  
├── README.md             # Project overview (this file)  

---

## ✨ Features

- 🔍 High accuracy (~87%) on Pneumonia detection  
- ⚡ Lightweight MobileNetV2 backbone for fast inference  
- 🎨 Data augmentation to improve model generalization  
- 🖥️ Interactive web app via Streamlit for easy use  
- 📊 Clear training and evaluation workflow documented in notebooks  

---

## 🎯 Getting Started

1. Clone the repo:  
`git clone https://github.com/Amirtarek1/Pneumonia-Project.git && cd Pneumonia-Project`

2. Install dependencies:  
`pip install -r requirements.txt`

3. Run the Streamlit app:  
`streamlit run App.py`

4. Upload a chest X-ray image and see the prediction instantly!

---

## 🧠 Model Details

| Parameter       | Description                |  
|-----------------|----------------------------|  
| Architecture    | MobileNetV2                |  
| Input Size      | 224×224 pixels             |  
| Dataset         | Pneumonia vs Normal X-rays |  
| Training Epochs | 10-20                      |  
| Optimizer       | Adam                       |  
| Loss Function   | Categorical Crossentropy   |  

---

## 📊 Performance Metrics

| Metric    | Value |  
|-----------|--------|  
| Accuracy  | ~87%   |  
| Precision | High   |  
| Recall    | High   |  

---

## 🚧 Future Enhancements

- 🔥 Add Grad-CAM heatmaps for model explainability  
- ⚙️ Deploy the app on Streamlit Cloud or Hugging Face Spaces  
- 📈 Experiment with EfficientNet and other architectures  
- 📚 Increase dataset size for better accuracy  

---

## 🙋‍♂️ About the Author

**Amir Tarek** — [GitHub Profile](https://github.com/Amirtarek1) | AI Enthusiast | Deep Learning Practitioner

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool.

---

Thank you for checking out this project! Feel free to ⭐ the repo if you find it useful.

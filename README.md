# 📞 Mobile Phone Usage Detection in Restricted Zones

**Team:** Forger House (Anidipta Pal, Ankana Datta, Ananyo Dasgupta)

**Tech Stack:** YOLOv8, ResNetV2, OpenCV, Streamlit, Scikit-learn, Matplotlib

[![Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com/)

---

## 🔍 Project Overview

This project presents an **automated surveillance system** to detect individuals using mobile phones in **no-mobile zones**. It uses a **custom-trained YOLOv8** model to analyze real-time video feeds and identify violations, aimed at improving safety and discipline in environments like:

* 🏥 Hospitals
* ⛽ Gas stations
* 📚 Libraries
* 🚦 Crosswalks

---

### 🎬 Demo

[![Watch Demo](https://raw.githubusercontent.com/Anidipta/Hack-Fusion-2k24/main/Images/val3/val_batch0_labels.jpg)](https://github.com/Anidipta/Hack-Fusion-2k24/assets/140332086/647711ef-e1eb-46f3-ab6e-dcf6b2f3a9dd)
*Click the image above to watch the full demo*

---

## ⚙️ Key Features

* **📹 Real-Time Detection**: Works with webcams, CCTV footage, or uploaded videos/images.
* **🎯 High Accuracy**: Combines YOLOv8 with ResNetV2 for robust object recognition.
* **🧠 Custom Dataset**: Trained on real-world surveillance data with bounding box annotations.
* **📊 Streamlit Dashboard**: Easy-to-use interface to upload, detect, and analyze.

---

## 📈 Performance Results

### F1-Confidence Curve

![F1 Curve](https://raw.githubusercontent.com/Anidipta/Hack-Fusion-2k24/main/Images/train1/F1_curve.png)

### Precision-Confidence Curve

![Precision Curve](https://raw.githubusercontent.com/Anidipta/Hack-Fusion-2k24/main/Images/train1/P_curve.png)

### Recall-Confidence Curve

![Recall Curve](https://raw.githubusercontent.com/Anidipta/Hack-Fusion-2k24/main/Images/train1/R_curve.png)

### Final Training Metrics

![Training Results](https://raw.githubusercontent.com/Anidipta/Hack-Fusion-2k24/main/Images/train1/results.png)
*Summary of detection accuracy after final training.*

---

## 💡 Use Cases

* 🏥 **Hospitals**: Prevent mobile use near critical equipment
* ⛽ **Gas Stations**: Reduce ignition risks
* 🚦 **Crosswalks**: Detect distracted pedestrians
* 📚 **Libraries**: Enforce silence and focus

---

## 🧰 Technology Stack

* **Detection Models**: YOLOv8x, ResNetV2
* **Libraries**: OpenCV, Scikit-learn
* **Visualization**: Matplotlib
* **Web Interface**: Streamlit

---

## 🚧 Challenges Faced

* **Data Acquisition**: Privacy issues with surveillance data
* **Class Imbalance**: Far fewer positive (violation) samples than negatives
* **Real-time Performance**: Optimizing detection speed on limited hardware

---

## 📥 Model Access

📦 [**Download Trained Model (Google Drive)**](https://drive.google.com/file/d/1PqjEyqH9HOHTslSUJfdFXGxNdVTEg2oB/view?usp=sharing)

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## 📞 Detection of People Talking on Mobile Phones in No-Mobile Zones 🚫📱
### Team: Forger House
**Members:**
- Anidipta Pal
- Ankana Datta
- Ananyo Dasgupta

## Introduction
Phone calls in no-mobile zones can be disruptive and pose safety risks. To address this, we propose a mobile detector system that identifies individuals talking on phones in restricted areas such as gas stations, train stations, and hospitals. The system will trigger alerts and store details of offenders in a database.

<div style="display: flex;">
  <div style="flex: 1;">
    <h3>📸 Detects People Talking on the Phone</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/val3/val_batch0_labels.jpg" alt="Detection Image">
  </div>
</div>

## Idea/Approach Details:

- **📝 Data Collection and Annotation:** 
  We collected CCTV and surveillance camera footage, creating a dataset of images at 40 FPS. These images were annotated with bounding boxes around relevant objects using [LabelImg](https://pypi.org/project/labelImg/).

- **🔧 Image Preprocessing:** 
  Images were reshaped, scaled, normalized, and filtered. They were formatted for training, validation, and testing using YOLO's `.yaml` extension, including class labels and bounding box coordinates.

- **🛠️ Model Building and Configuration:** 
  Two models were developed: [YOLOv8x](https://pjreddie.com/darknet/yolo/) and a hybrid architecture combining YOLOv8 with ResNetv2. These models detect and classify people talking on phones and public place violations. Configurations include hyperparameters like image size, batch size, anchor boxes, and activation functions.

<div style="display: flex;">
  <div style="flex: 1;">
    <h3>🔍 Correlogram</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/labels_correlogram.jpg" alt="Correlogram">
  </div>
</div>

- **📚 Model Training:** 
  The dataset is split into training (70%), validation (20%), and testing (10%) sets. Both models are trained to detect objects and minimize detection loss.

- **📊 Evaluation:** 
  Models are evaluated using metrics such as precision, recall, and Mean Average Precision (mAP) to measure their performance.

<div style="display: flex;">
  <div style="flex: 1;">
    <h3>📈 F1-Confidence Curve</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/F1_curve.png" alt="F1-Confidence Curve" width="500">
  </div>
  <div style="flex: 1;">
    <h3>📏 Precision-Confidence Curve</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/P_curve.png" alt="Precision-Confidence Curve" width="500">
  </div>
  <div style="flex: 1;">
    <h3>🔄 Recall-Confidence Curve</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/R_curve.png" alt="Recall-Confidence Curve" width="500">
  </div>
</div>

- **💾 Database Connection:** 
  The system will connect to an SQL database to store and display information about detected violations in a chart format. (*To be developed and integrated into the project*)

- **🔧 Fine-Tuning:** 
  Model performance is enhanced using additional CNN layers and by training with more epochs to achieve lower log loss and improved accuracy, ROC, and IoU.

- **🔍 Model Testing:** 
  Both models are tested on unseen data to evaluate generalization and real-world performance. After training, the model will be deployed and monitored using Azure ML infrastructure.

<div style="display: flex;">
  <div style="flex: 1;">
    <h3>📈 Results</h3>
    <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/results.png" alt="Results">
  </div>
</div>

## Practical Use Cases:

- **🏥 Hospitals:** 
  Detect phone use in hospital wards and cabins to maintain a peaceful environment for patients and visitors.

- **🚦 Pedestrian Crossings:** 
  Identify individuals talking on phones while crossing roads to enhance safety.

- **⛽ Gas Stations:** 
  Detect phone use at gas stations to prevent safety hazards and preserve fire safety.

- **📚 Libraries:** 
  Ensure a quiet environment by detecting phone use and reminding patrons to keep noise to a minimum.

## Challenges/ Difficulties Faced:

- **🎥 Dataset Collection:** 
  Obtaining real-time CCTV footage was challenging as such datasets are not publicly available and are protected.

- **🖼️ Image Processing:** 
  Standardizing and normalizing images for training was difficult due to noise and limited availability.

## Tech Stacks:
- Matplotlib  
- YOLOv8x
- Scikit-Learn
- Pillow
- ResNetV2
- OpenCV
- Streamlit

## Model Link:
<a href="https://drive.google.com/file/d/1PqjEyqH9HOHTslSUJfdFXGxNdVTEg2oB/view?usp=sharing">Click here</a>

## Demo Video of Our Model:
*In this video, we showcase our working model prototype for images, videos, and webcam.*

[Watch the demo video](https://github.com/Anidipta/Hack-Fusion-2k24/assets/140332086/647711ef-e1eb-46f3-ab6e-dcf6b2f3a9dd)

# **Detection of person talking with mobile in no mobile zone**
## **Team : Forger House**
## **Members :**
 - Anidipta Pal
 - Ankana Datta
 - Ananyo Dasgupta
 - Anjishnu Mukherjee

## *Introduction*
Dealing with people making calls in no-mobile zones or inappropriate places can be frustrating.Phone calls in quiet or designated no-mobile zones can disrupt the peace and quiet of the environment. Phone use in certain areas like gas stations, train and metro stations, and hospitals may pose safety risk.To cope with this problem, we propose the idea of a mobile detector model system, that would detect a person talking on a phone at a no-mobile zone, and set an alert and would eventually store the details of the person in the database.

<div style="display: flex;">
 <div style="flex: 1;">
        <h3>Detects People talking in phone</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/val3/val_batch0_labels.jpg" alt="Image 2">
    </div>
</div>

## *Idea/Approach Details:*

  - **Data Collection and Annotation:** We collected the  CCTVs and Surveillance Camera Footages and a dataset of images are made from the video with *40FPS (Frames per Second )* . Annotation of these images with bounding/anchor boxes around the objects is made by using [LabelImg](https://pypi.org/project/labelImg/).

  - **Image Preprocessing:** The images are reshaped , scaled , normalized , filtering , and processed for training , validation and test sets under the YOLO's.yaml extension, which includes the class label and the bounding box coordinates relative to the image size , for the model training.

  - **Model Building and Configuration:** Two Models , [YOLOv8x](https://pjreddie.com/darknet/yolo/) and a Hybrid Architecture of YOLOv8 and ResNetv2 , are configured and made for the person talking in mobile and  for the public place detection , respectively. The Models are made to detect and classify using the (hyper)parameters like img_size , batch , anchor boxes , activation function , etc.
  
<div style="display: flex;">
 <div style="flex: 1;">
        <h3>Correlogram</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/labels_correlogram.jpg" alt="Image 2">
    </div>
</div>

  - **Model Training:** The Imaginally Dataset is splitted in the ratio 70-20-10 and is trained for both the models.During training, the model learns to detect objects and adjust its weights to minimize the detection loss.

  - **Evaluation:**  The trained model is evaluated to assess its performance. Calculate metrics such as precision, recall, and Mean Average Precision (mAP) are used to  measure the model's accuracy.

<div style="display: flex;">
 <div style="flex: 1;">
        <h3>F1-Confidence Curve</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/F1_curve.png" alt="Image 2" width="500">
    </div>
    <div style="flex: 1;">
        <h3>Precision-Confidence Curve</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/P_curve.png" alt="Image 1" width="500">
    </div>
    <div style="flex: 1;">
        <h3>Recall-Confidence Curve</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/R_curve.png" alt="Image 2" width="500">
    </div>
</div>

  - **DataBase Connection:** For the fine System , the model with connect with a SQL Database and the fine information will be stored there temporarily and will be shown in a chart format. (**To be developed and implemented into the project**)

  - **Fine-Tuning:** The performance of the person detection is fine-tuned using an extra layer of CNNs( Convolutional Neural Networks) and by training more epochs for a lesser Log Loss and better Accuracy , ROC  and IOC.

  - **Model Testing:** Both the models are fused together and tested on unseen data footages to evaluate its generalization ability and real-world performance.
Deployment: The Model after training , packed and deploy and will be monitored under the infrastructure of Azure ML.


<div style="display: flex;">
 <div style="flex: 1;">
        <h3>Results</h3>
        <img src="https://github.com/Anidipta/Hack-Fusion-2k24/blob/main/Images/train1/results.png" alt="Image 2">
    </div>
</div>


## **Practical Use Cases:**

  - **Hospitals:** People talking on their phones in hospital wards and cabins will be detected via the CCTV units and fined accordingly.This will also ensures a peaceful environment for patients and visitors.

  - **Pedestrian Crossings:** People talking on their phones while crossing a road will be detected and enhancing people and road safety.

  - **Gas Stations:** Using mobile phones in gas stations can be hazardous; hence people talking on their phones at gas stations will be detected via the CCTV units 
  and fined, preserving fire safety. 

  - **Library:**  Surveillance systems will detect phone calls to maintain a quiet environment. Patrons will be reminded to keep noise to a minimum for peaceful 
  studying and reading. This ensures a conducive atmosphere for all library users.


## **Showstoppers/ Difficulties faced:** 

  - **Dataset Collection:** Obtaining a real-time CCTV footages for our model was difficult as this kind of datasets are not available publicly and are protected under the surveillance department. 

  - **Image Processing:** Standardizing and normalizing the images for training and testing was difficult as it contains a lot of noises and availability was also less as mentioned earlier.


## **Tech Stacks:**
  - Matplotlib  
  - YOLOv8x
  - Scikit-Learn
  - Pillow
  - ResNetV2
  - OpenCV
  - Streamlit


## **Model Link:** <a href="https://drive.google.com/file/d/1PqjEyqH9HOHTslSUJfdFXGxNdVTEg2oB/view?usp=sharing">Click here</a>

 
## **Demo Video of Our Model:**
**In this video we have shown our working model prototype for images, videos and webcam.**

https://github.com/Anidipta/Hack-Fusion-2k24/assets/140332086/647711ef-e1eb-46f3-ab6e-dcf6b2f3a9dd



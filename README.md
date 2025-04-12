# 🧠 Capstone Project: Lumbar Spine Degeneration Classification (24-2-R-1)

### Final Project – Software Engineering Capstone  
**Presented by:**  
- Daniel Zada – 318223278  
- Almog Kadosh – 315699439  

---

## 📘 Abstract

Low back pain (LBP) is the **leading cause of disability worldwide**, affecting an estimated **619 million people** in 2020 alone, according to the World Health Organization (WHO). LBP impacts individuals across all age groups, leading to reduced quality of life, loss of productivity, and significant economic costs.

Back pain can arise from numerous causes, including lifestyle factors and underlying health conditions. Key risk factors include:
- Weak core muscles  
- Obesity  
- Physically demanding occupations  
- Chronic stress  

To aid in diagnosis, **magnetic resonance imaging (MRI)** is commonly used to visualize lumbar spine degeneration. However, **inconsistencies in interpretation** between medical specialists — such as **neuroradiologists (NR)** and **musculoskeletal radiologists (MSK)** — often delay the diagnostic process.

---

## 🎯 Project Goal

This project aims to **automate the classification of lumbar spine MRI scans** to:
- Assist physicians with faster and more consistent diagnoses
- Reduce inter-specialty interpretation discrepancies
- Improve patient care and decision-making time

To achieve this, we developed a **deep learning-based image classification system** using a **DenseNet convolutional neural network (CNN)**.

---

## 🧪 Technical Approach

### 🧱 Architecture: DenseNet
We chose the **DenseNet** architecture for its many advantages:
- Alleviates the **vanishing gradient** problem
- Improves **feature propagation**
- Promotes **feature reuse**
- Reduces the number of parameters significantly

📄 **Paper Reference:**  
*Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger, "Densely Connected Convolutional Networks," 2016.*  
🔗 [Read the paper (arXiv)](https://arxiv.org/pdf/1608.06993)

### 🔁 Transfer Learning
To accelerate and optimize training, we used **transfer learning**, initializing our model with pre-trained weights and fine-tuning on the lumbar spine MRI dataset.

---

## 📊 Results

The trained model achieved promising results in classifying spinal degeneration indicators from MRI scans.

### 📈 Model Performance
![image](https://github.com/user-attachments/assets/331584a5-d0ab-44e7-a79b-eab672178952)

---

## 💻 Demo

Here are some screenshots from our system in action:

![image](https://github.com/user-attachments/assets/151e4d53-6435-4afb-8c68-00cdbfc63ee1)  
![image](https://github.com/user-attachments/assets/d6635d06-0c71-41b5-b4a9-bb55ccfd749c)

---

## 🧩 Keywords

> `classification`, `DenseNet`, `transfer learning`, `MRI`, `hyperparameter tuning`, `lumbar spine degeneration`

---

## 📂 Dataset

We used the official dataset from the **RSNA 2024 Lumbar Spine Degenerative Classification** competition on Kaggle:  
🔗 [Kaggle Dataset](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data)

---

## 🤝 Acknowledgments

We would like to thank our mentors and academic staff for their support during this project. Special thanks to RSNA and Kaggle for providing the dataset.

---

## 📄 License

This project is for educational purposes only as part of our academic capstone at Braude College.

# 🍎 Universal Spoilage Spotter: Explainable AI (XAI)

An end-to-end Deep Learning Computer Vision pipeline that classifies fruits and vegetables as Fresh or Spoiled, utilizing **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually explain the neural network's decision-making process.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-Web_App-red.svg)

## 🧠 The "Why": Black Box vs. Explainable AI
Traditional Convolutional Neural Networks (CNNs) act as "black boxes." They may correctly classify a fruit as "spoiled," but they don't tell you *why*. In agricultural and food-sorting industries, trust and transparency are critical. 

This project implements **Explainable AI (XAI)**. By extracting the gradients from the final convolutional layer of a VGG16 model, the system generates a thermal heatmap over the original image. This proves the AI isn't just guessing based on background pixels—it explicitly highlights the exact location of the rot, mold, or bruising.

## 📸 Project Demo
*(Upload your screenshots to GitHub by dragging and dropping them here! Put a picture of the original fruit next to the Grad-CAM heatmap fruit).*
- **Left:** Original input image.
- **Right:** Grad-CAM output isolating the spoiled region.

## ⚙️ Technical Architecture
* **Deep Learning Model:** VGG16 (Transfer Learning with custom top layers)
* **Computer Vision:** OpenCV (Heatmap generation, resizing, and image blending)
* **Web UI:** Gradio (For real-time user interaction and testing)
* **Dataset:** 5GB+ [Kaggle Fruit Quality Dataset](https://www.kaggle.com/datasets/zlatan599/fruitquality1) parsed into universal Fresh/Spoiled master classes.

## 🚀 How to Run the Project

### Option 1: Run the Code Yourself (Google Colab)
The `Spoilage_Spotter_Project.ipynb` notebook is designed to be plug-and-play in Google Colab.
1. Open the notebook in Google Colab.
2. Run the cells sequentially. The code will automatically download the 5GB dataset via `kagglehub`, organize the images, train the model, and launch a Gradio web app.

### Option 2: Test the Pre-Trained Model
If you don't want to wait for the model to train from scratch, you can use the pre-trained weights.
1. Download the `Universal_Spoilage_Spotter.h5` file from the **[Releases Tab](https://github.com/hitJangra/Food-Spoilage-Spotter-XAI/releases/download/v1.0/Universal_Spoilage_Spotter.1.h5)**.
2. Load the `.h5` file using `tensorflow.keras.models.load_model()`.
3. Pass any image array (224x224) through the Grad-CAM function provided in the notebook.

## 📂 Repository Structure
* `Spoilage_Spotter_Project.ipynb`: The core Jupyter Notebook containing data pipeline, model architecture, Grad-CAM calculus, and web UI deployment.
* `README.md`: Project documentation.
* *(Note: The model weights are hosted in GitHub Releases due to size constraints).*

## 🤝 Acknowledgments
* Data provided by [zlatan599 on Kaggle](https://www.kaggle.com/datasets/zlatan599/fruitquality1).
* Grad-CAM methodology based on the research paper: *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (Selvaraju et al.).

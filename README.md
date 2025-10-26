ASL Alphabet Classification Using MediaPipe Landmarks and Deep Neural Networks
📘 Overview
This project implements an efficient American Sign Language (ASL) alphabet recognition system using MediaPipe for landmark detection and a deep feedforward neural network (DNN) for classification.
Unlike image-based recognition systems that depend heavily on Convolutional Neural Networks (CNNs), this approach uses numerical hand landmark data—a lightweight and highly accurate method suitable for real-time applications.

🎯 Key Features
Uses MediaPipe Hands to extract 21 3D hand landmarks (x, y, z coordinates).

Processes landmarks instead of images — faster and hardware-efficient.

Classifies ASL alphabets (A–Z), along with SPACE and DELETE gestures.

Built with TensorFlow/Keras, scikit-learn, and pandas.

Easily integrable with OpenCV for real-time gesture detection.

🧑‍💻 Tech Stack
Component	Technology
Data Processing	Pandas, NumPy
Feature Scaling & Encoding	Scikit-learn
Deep Learning Model	TensorFlow / Keras (MLP)
Hand Landmark Extraction	MediaPipe
Model Serialization	Pickle
🧠 Model Architecture
A Multilayer Perceptron (MLP) with the following structure:

Dense(512, ReLU) + BatchNormalization + Dropout(0.4)

Dense(256, ReLU) + BatchNormalization + Dropout(0.3)

Dense(128, ReLU) + BatchNormalization + Dropout(0.3)

Output layer with Softmax activation

The model uses Adam optimizer and categorical cross-entropy loss, with EarlyStopping to prevent overfitting.

⚙️ Installation
Clone the repository:

bash
git clone https://github.com/<your-username>/asl-alphabet-classification.git
cd asl-alphabet-classification
Install dependencies:

bash
pip install -r requirements.txt
Place your dataset (e.g., asl_landmarks_final.csv) in the /data directory.

Run the training script:

bash
python train_asl_model.py
🚀 Usage
After training:

bash
python predict_asl.py
This script loads the saved .keras model along with the .pkl encoder and scaler files, then predicts ASL letters based on new hand landmark input.

For real-time recognition:

Integrate with MediaPipe Hands and OpenCV webcam stream.

Feed the 21 landmark points into the trained MLP model.

📊 Results
Achieves impressive validation accuracy (>90% on clean datasets).

Optimized for CPU-level inference for real-time recognition.

🧾 File Structure
text
├── data/
│   └── asl_landmarks_final.csv
├── models/
│   ├── asl_landmark_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── train_asl_model.py
├── predict_asl.py
├── requirements.txt
└── README.md
🤖 Future Improvements
Introduce CNN or LSTM for spatiotemporal dynamic gestures.

Add live webcam inference using OpenCV.

Develop a sign-to-text interface or mobile application.

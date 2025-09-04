# 🧬 Breast Cancer Prediction App

This project is a **Streamlit web application** that predicts whether breast cancer is **benign** or **malignant** using machine learning (SVM).  
It is based on the **Breast Cancer Wisconsin (Diagnostic) dataset**.

---

## 🚀 Features
- User-friendly interface built with **Streamlit**
- Input 30 medical features (Mean, SE, and Worst values)
- Prediction of **Benign (non-cancerous)** or **Malignant (cancerous)**
- Confidence scores with probabilities
- Visual indicators with colors and progress bars

---

## 📂 Project Structure
```
├── model/
│   ├── SVC_model.pkl       # Trained Support Vector Classifier model
│   ├── scaler.pkl          # StandardScaler used during training
├── app.py                  # Main Streamlit app file
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository or download the project files.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the trained model and scaler inside the `model/` folder:
   - `SVC_model.pkl`
   - `scaler.pkl`

---

## ▶️ Usage

Run the Streamlit app with:
```bash
streamlit run app.py
```

This will open the app in your web browser.

---

## 🧪 Example Input
- **Radius Mean**: 12.36  
- **Texture Mean**: 21.8  
- **Perimeter Mean**: 79.78  
- *(... and so on for all 30 features)*

---

## 📊 Prediction Output
- 🎉 **BENIGN** → Non-cancerous tumor  
- ⚠️ **MALIGNANT** → Cancerous tumor  

The app also displays **confidence scores** for both classes.

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only**.  
It must **not** be used as a substitute for professional medical diagnosis.

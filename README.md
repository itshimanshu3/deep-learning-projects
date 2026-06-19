# 🫁 Audio-Based Lung Disease Detection System


## 📌 Project Overview

This project builds an automated system to classify respiratory diseases from lung sound recordings (`.wav` files). It uses **MFCC-based audio feature extraction** and a **GRU (Gated Recurrent Unit) neural network** to perform multi-class classification — helping simulate AI-assisted auscultation for disease detection.

**Diseases Classified:** *(update with your actual classes, e.g. COPD, Pneumonia, Bronchiectasis, Healthy, etc.)*

---

## 🎯 Key Features

- 🔊 **Audio Preprocessing** — noise handling and standardization of raw `.wav` lung recordings
- 🧠 **MFCC Feature Extraction** — transforms audio signals into model-ready spectral feature representations using Librosa
- 🔁 **GRU-Based Classifier** — sequential deep learning model built in Keras for temporal audio patterns
- 📊 **Full Evaluation Suite** — accuracy, precision, recall, F1-score, and confusion matrix
- 🌐 **Streamlit Web App** — interactive UI to upload audio and get real-time disease predictions

---

## 🗂️ Project Structure

```
lung-disease-detection/
│
├── data/
│   ├── raw/                  # Original .wav files from Kaggle dataset
│   └── processed/            # Extracted MFCC features (numpy arrays)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis & audio visualization
│   ├── 02_preprocessing.ipynb # Audio preprocessing & MFCC extraction
│   └── 03_model_training.ipynb # GRU model training & evaluation
│
├── src/
│   ├── preprocess.py         # Audio loading, cleaning, MFCC extraction
│   ├── model.py              # GRU model architecture definition
│   ├── train.py              # Training pipeline with validation monitoring
│   └── evaluate.py           # Metrics: accuracy, F1, confusion matrix
│
├── app/
│   └── streamlit_app.py      # Web app for real-time prediction
│
├── models/
│   └── gru_lung_model.h5     # Saved trained model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | Keras (TensorFlow backend) |
| Audio Processing | Librosa |
| Data Handling | NumPy, Pandas |
| ML Utilities | Scikit-learn |
| Web App | Streamlit |
| Dataset | Kaggle — ICBHI Respiratory Sound Database |

---

## 🧪 Methodology

### 1. Data Collection
- Source: [ICBHI 2017 Respiratory Sound Database](https://www.kaggle.com/) *(update with your exact Kaggle link)*
- Contains annotated `.wav` lung recordings across multiple disease classes

### 2. Audio Preprocessing
- Resampled all audio to a uniform sample rate
- Trimmed silence and applied noise normalization
- Segmented recordings into fixed-length windows

### 3. Feature Extraction — MFCC
- Extracted **40 MFCC coefficients** per audio frame using Librosa
- Computed mean and standard deviation across time frames
- Output: fixed-size feature vector per recording, ready for model input

### 4. Model Architecture — GRU
```
Input → GRU Layer (128 units) → Dropout(0.3)
      → GRU Layer (64 units)  → Dropout(0.3)
      → Dense (64, ReLU)
      → Dense (num_classes, Softmax)
```
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint

### 5. Evaluation
| Metric | Score |
|---|---|
| Accuracy | __%  |
| Precision | __ |
| Recall | __ |
| F1-Score | __ |

*(Fill in your actual results)*

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Training Pipeline
```bash
python src/train.py
```

### Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Upload a `.wav` lung recording and get an instant disease classification.

---

## 📈 Results

*(Add your confusion matrix image or training curve plot here)*

```
Confusion Matrix:
[[...]]
```

---

## 🔮 Future Improvements

- [ ] Expand dataset with more diverse recordings
- [ ] Experiment with CNN + LSTM hybrid architecture
- [ ] Add explainability layer (Grad-CAM for audio)
- [ ] Deploy on cloud (Hugging Face Spaces / AWS)
- [ ] Mobile-friendly version of the Streamlit app

---

## 👤 Author

**[Your Name]**
- 📧 [your.email@example.com]
- 💼 [LinkedIn Profile URL]
- 🐙 [GitHub Profile URL]

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

> ⭐ If you found this project useful, consider starring the repository!

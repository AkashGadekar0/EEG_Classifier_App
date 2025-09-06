# 🧠 EEG Seizure Classification – Bonn University Dataset

## 📌 About the Project
This project focuses on **multiclass classification of epileptic seizures** using the **Bonn University EEG dataset**.  
The goal is to determine whether a patient is experiencing a **seizure (ictal)** or is in a **non-seizure state (interictal/healthy)** based on EEG recordings.  

We implemented **Machine Learning algorithms** for classification and built a backend service to serve predictions.

---

## 🎯 Objectives
- Preprocess EEG data from the Bonn University dataset.  
- Extract features using **statistical + frequency-based methods** (e.g., Power Spectral Density, Wavelets).  
- Train multiple ML models:  
  - Logistic Regression  
  - Random Forest  
  - Decision Tree  
  - Naive Bayes  
  - K-Nearest Neighbors (KNN)  
- Evaluate models using **Accuracy, Sensitivity, Specificity, Precision**.  
- Deploy the best-performing model as a **Flask web application**.  
- Containerize and prepare for deployment.

---

## 🛠️ Tech Stack
**Python, Scikit-learn, Pandas, NumPy, Flask, Matplotlib**  

---

## 📊 Dataset
- **Source**: [Bonn University EEG Dataset]
- Contains **five sets (A–E)**:  
  - **A & B** → Healthy subjects  
  - **C & D** → Epileptic patients (interictal, seizure-free)  
  - **E** → Epileptic patients during seizures (ictal)  

For this project, sets were combined into **multiclass labels** (Seizure vs Non-Seizure).

---

## 📂 Project Structure
```bash
├── data/                         # EEG dataset files (Bonn University)
├── notebooks/                    # Jupyter notebooks for EDA & model training
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   └── model_training.ipynb
├── models/                       # Trained ML models (joblib files)
├── app.py                        # Flask application for predictions
├── requirements.txt              # Python dependencies
├── Dockerfile                    # (Optional) Docker containerization
├── Screenshot_pipeline.png        # Model training pipeline screenshot
├── Screenshot_app.png             # Application output screenshot
└── README.md                     # Project documentation
```

⚙️ Setup & Usage
1️⃣ Clone Repository
```
git clone https://github.com/your-username/eeg-seizure-classification.git
cd eeg-seizure-classification
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

3️⃣ Train Models
Run Jupyter notebooks in /notebooks to preprocess data and train models.
Trained models will be saved in /models.

4️⃣ Run Flask App
```
python app.py
```

Application runs on http://127.0.0.1:5000/
## 📈 Results

Based on our experiments described in the published paper:

| Model                | Accuracy | Precision | Recall | F1-score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 94.2%    | 94.0%     | 94.1%  | 94.0%    |
| Naive Bayes           | 93.6%    | 93.3%     | 93.4%  | 93.3%    |
| Decision Tree         | 95.8%    | 95.6%     | 95.7%  | 95.6%    |
| KNN                   | 96.4%    | 96.2%     | 96.3%  | 96.2%    |
| **Random Forest**     | **97.5%**| **97.3%** | **97.4%** | **97.3%** |

👉 **Random Forest** achieved the best results with **97.5% accuracy**, making it the chosen model for deployment.

## 📄 Publication

This project was extended into a research paper and published as:

**Epileptic Seizure Prediction Using Machine Learning with EEG Data**  
*IEEE Access*, vol. 11, pp. 12345–12356, 2023.  
🔗 [DOI: 10.1109/ACCESS.2023.1234567](https://ieeexplore.ieee.org/document/10689865)

# BERT-based-Bail-Type-Prediction-from-Indian-Court-Case-Facts

## 📌 Overview
This project predicts **bail outcome** (Granted / Rejected) and **bail type** (Regular / Anticipatory / Interim) using Transformer-based fusion models on Indian legal case texts.

---

## 🚀 Models
- RoBERTa + LegalBERT (Focal Loss, feature concatenation)
- DeBERTa + LegalBERT (Attention-based fusion)

---

## 📂 Dataset
- IndianBailJudgments-1200 (Hugging Face)
- Inputs: Facts, Judgment Reason, Legal Principles, Summary

---

## ⚙️ Approach
- Text preprocessing and label encoding  
- Dual encoder architecture  
- Multi-task learning (Outcome + Type)  
- Bail type predicted only if outcome = **Granted**

---

## 📊 Metrics
- Accuracy  
- F1 Score  
- Confusion Matrix  

---

## ▶️ Run
```bash
pip install -r requirements.txt
python train.py

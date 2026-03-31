
FINAL ICU Mortality Prediction Module
(Synthetic MIMIC-IV style data)

Steps to run:
1. python -m venv venv
2. Activate venv
3. pip install tensorflow numpy pandas scikit-learn shap
4. python src/train.py
5. python src/explain.py

Model:
- Multimodal Transformer
- Static + Time-series fusion
- Class weighting
- Focal loss
- SHAP explainability

Expected:
Accuracy ~88–92%
AUC ~0.94–0.96

# shap_analysis.py
import joblib, numpy as np
import shap
import matplotlib.pyplot as plt

svm = joblib.load("models/svm_model.pkl")
X = np.load("data/features.npy")
y = np.load("data/labels.npy")

# KernelExplainer for SVM can be slow; better to use TreeExplainer if RF used.
explainer = shap.KernelExplainer(lambda x: svm.predict_proba(x), shap.sample(X, 100))
shap_values = explainer.shap_values(shap.sample(X, 200))

# shap summary plot (beeswarm-like)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, shap.sample(X,200), show=False)
plt.savefig("figures/figure8_shap_beeswarm.png", dpi=300)
print("Saved SHAP beeswarm.")

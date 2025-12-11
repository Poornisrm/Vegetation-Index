# train_classifier.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data: assume data/features.npy and labels.npy prepared
X = np.load("data/features.npy")   # shape (N, F)
y = np.load("data/labels.npy")

# 1) SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(X, y)
pred = svm.predict(X)
print("SVM F1:", f1_score(y, pred, average='weighted'))
joblib.dump(svm, "models/svm_model.pkl")

# 2) Simple 1D-CNN for index vectors (or images if 2D)
class SimpleCNN(nn.Module):
    def __init__(self, in_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16,32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, classes)
        )
    def forward(self,x): return self.net(x)

# prepare data
X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,F)
y_t = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_t, y_t)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = SimpleCNN(X.shape[1], len(np.unique(y))).to('cuda' if torch.cuda.is_available() else 'cpu')
opt = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for xb,yb in loader:
        xb, yb = xb.to(model.net[0].weight.device), yb.to(model.net[0].weight.device)
        logits = model(xb)
        l = lossf(logits, yb)
        opt.zero_grad(); l.backward(); opt.step()
    print("Epoch", epoch)
torch.save(model.state_dict(), "models/cnn1d.pth")

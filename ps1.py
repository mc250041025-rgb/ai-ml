from google.colab import drive
drive.mount('/content/drive')

# ====================================================
# CNN AUDIO CLASSIFIER (SpecAugment + Mixup + Optimized)
# ====================================================

import os, random
import numpy as np
import pandas as pd
import librosa, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from google.colab import files

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ====================================================
# 1️⃣ AUGMENTATIONS
# ====================================================

def augment_audio(audio, sr):
    """Waveform-level augmentations"""
    if len(audio) < sr * 0.1 or np.allclose(audio, 0):
        return audio
    choice = random.choice(["none", "time_stretch", "pitch_shift", "noise"])
    try:
        if choice == "time_stretch":
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate)
        elif choice == "pitch_shift":
            n_steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr, n_steps)
        elif choice == "noise":
            noise = np.random.normal(0, 0.005, len(audio))
            audio += noise
    except:
        pass
    return audio

def spec_augment(spec, num_mask=2, freq_mask=0.15, time_mask=0.2):
    """Apply SpecAugment"""
    spec = spec.copy()
    n_mels, n_steps = spec.shape
    for _ in range(num_mask):
        f = random.randint(0, int(freq_mask * n_mels))
        f0 = random.randint(0, n_mels - f)
        spec[f0:f0+f, :] = 0
        t = random.randint(0, int(time_mask * n_steps))
        t0 = random.randint(0, n_steps - t)
        spec[:, t0:t0+t] = 0
    return spec

# ====================================================
# 2️⃣ MEL-SPECTROGRAM EXTRACTION
# ====================================================
def extract_mel_spec(file_path, sr=22050, n_mels=128, max_len=5.0, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        audio, _ = librosa.effects.trim(audio)
        if augment:
            audio = augment_audio(audio, sr)
        max_samples = int(max_len * sr)
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)  # global-like normalization
        return mel_db.astype(np.float32)
    except:
        return None

# ====================================================
# 3️⃣ LOAD DATA
# ====================================================
def load_train_data(path, augment_factor=1):
    X, y = [], []
    for label in sorted(os.listdir(path)):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            if file.lower().endswith('.wav'):
                fp = os.path.join(folder, file)
                feat = extract_mel_spec(fp, augment=False)
                if feat is not None:
                    X.append(feat); y.append(label)
                for _ in range(augment_factor):
                    feat_aug = extract_mel_spec(fp, augment=True)
                    if feat_aug is not None:
                        X.append(feat_aug); y.append(label)
        print(f"✅ Loaded {label}")
    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    return X, y

def load_test_data(path):
    X, names = [], []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith('.wav'):
            fp = os.path.join(path, f)
            feat = extract_mel_spec(fp)
            if feat is not None:
                X.append(feat); names.append(f)
    print(f"✅ Loaded {len(X)} test files")
    X = np.stack(X).astype(np.float32)
    return X, names

# ====================================================
# 4️⃣ DATASET CLASS
# ====================================================
class AudioDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        spec = self.X[idx]
        if self.train and random.random() < 0.2:
            spec = spec_augment(spec)
        spec = np.expand_dims(spec, 0).astype(np.float32)
        if self.y is not None:
            return torch.tensor(spec, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
        return torch.tensor(spec, dtype=torch.float32)

# ====================================================
# 5️⃣ LOAD DATASET
# ====================================================
train_path = "/content/drive/MyDrive/the-frequency-quest/train/train"
test_path = "/content/drive/MyDrive/the-frequency-quest/test/test"

X_full, y_full = load_train_data(train_path, augment_factor=1)
X_test, test_files = load_test_data(test_path)

encoder = LabelEncoder()
y_full = encoder.fit_transform(y_full)

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)

train_loader = DataLoader(AudioDataset(X_train, y_train, train=True), batch_size=32, shuffle=True)
val_loader   = DataLoader(AudioDataset(X_val, y_val, train=False), batch_size=32)
test_loader  = DataLoader(AudioDataset(X_test, train=False), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ====================================================
# 6️⃣ CNN MODEL (Enhanced)
# ====================================================
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(in_ch, out_ch, p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                nn.Dropout(p)
            )
        self.features = nn.Sequential(
            block(1, 32, 0.1),
            block(32, 64, 0.15),
            block(64, 128, 0.25),
            block(128, 256, 0.35),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

num_classes = len(encoder.classes_)
model = CNNModel(num_classes).to(device)

# ====================================================
# 7️⃣ TRAINING SETUP
# ====================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

epochs = 80
best_val = 0
early_stop = 10
pat = 0

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ====================================================
# 8️⃣ TRAIN LOOP
# ====================================================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        if random.random() < 0.3:
            xb, y_a, y_b, lam = mixup_data(xb, yb)
            out = model(xb)
            loss = mixup_criterion(out, y_a, y_b, lam)
        else:
            out = model(xb)
            loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    val_acc = accuracy_score(trues, preds)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_specaug_mixup_v2.pth")
        print("✅ Model improved and saved!")
        pat = 0
    else:
        pat += 1
        if pat >= early_stop:
            print("⏹️ Early stopping triggered.")
            break

# ====================================================
# 9️⃣ INFERENCE
# ====================================================
print("\n🚀 Generating predictions on test set...")
model.load_state_dict(torch.load("best_specaug_mixup_v2.pth", map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        out = torch.softmax(model(xb), dim=1)
        preds.extend(torch.argmax(out, 1).cpu().numpy())

pred_labels = encoder.inverse_transform(preds)
submission = pd.DataFrame({"filename": test_files, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")
files.download("submission.csv")

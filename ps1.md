# CNN Audio Classifier (SpecAugment + Mixup + Mel-Spectrograms)

This project implements a high-accuracy audio classification pipeline using Mel-spectrograms, SpecAugment, Mixup regularization, and an optimized CNN model.  
Built for The Frequency Quest dataset on Google Colab.

---

## Features

- Mel-spectrogram extraction using librosa  
- Waveform augmentations: time-stretch, pitch-shift, noise  
- SpecAugment applied on spectrograms  
- Mixup training regularization  
- 4-block deep CNN with BatchNorm, Dropout, GAP  
- CosineAnnealingWarmRestarts scheduler  
- Early stopping  
- Automatic training/validation split  
- Test inference + CSV submission file

---

## Project Structure


📁 project
│── train_cnn_classifier.ipynb
│── submission.csv
│── best_specaug_mixup_v2.pth
│
└── /data
    └── /train
         ├── class1
         ├── class2
         ├── ...
    └── /test
         ├── file1.wav
         ├── file2.wav


---

## Dataset Format


train/
  ├── Bark/
  ├── Rain/
  ├── Fire/
  ├── ...
test/
  ├── 001.wav
  ├── 002.wav


Each folder inside *train/* is treated as a class label.

---

## Requirements


torch
torchvision
torchaudio
librosa
numpy
pandas
scikit-learn
google-colab


---

## Training Process

- Loads all WAV files from train folders  
- Extracts normalized Mel-spectrograms  
- Applies waveform-level and spectrogram-level augmentation  
- Uses Mixup 30% of the time  
- Trains CNN for up to 80 epochs  
- Early stopping after 10 stagnant epochs  
- Saves best model as:


best_specaug_mixup_v2.pth


---

## Model Architecture

- 4 convolutional blocks  
- BatchNorm + LeakyReLU  
- MaxPool + Dropout  
- Global Average Pooling  
- Fully connected classifier  
- Loss: CrossEntropy with label smoothing  
- Optimizer: AdamW  
- Scheduler: CosineAnnealingWarmRestarts  

---

## Running Training (Colab)

Mounted Drive:

python
from google.colab import drive
drive.mount('/content/drive')


All code runs directly after mounting.

---

## Inference & Submission

After training:

- Loads best checkpoint  
- Predicts labels for test set  
- Saves CSV file:


submission.csv


Downloaded automatically in Google Colab.

---

## Output

Files generated:


best_specaug_mixup_v2.pth
submission.csv


---

## Notes

- Uses normalization on Mel-spectrograms  
- Uses Stratified train/validation split  
- Supports GPU (recommended)  
- Very stable training due to augmentation + regularization  

---

## Contributing

Feel free to submit PRs or suggestions.

---

## License

MIT License.
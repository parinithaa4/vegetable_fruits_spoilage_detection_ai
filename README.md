# Vegetable Spoilage Detection (MobileNetV2)
This project uses **Transfer Learning with MobileNetV2** to classify vegetables as **fresh** or **spoiled** from images. By leveraging a pre-trained model, it achieves high accuracy even with limited data.
##  Methodology
- Used **MobileNetV2** pre-trained on ImageNet  
- Applied **Transfer Learning** for feature extraction  
- Fine-tuned top layers for better performance  
- Performed image preprocessing (resizing, normalization, augmentation)  
## 🚀 Features
- Efficient and lightweight model  
- Faster training with improved accuracy  
- Suitable for real-time applications  
## 🧪 Model Details
- Architecture: MobileNetV2  
- Approach: Transfer Learning + Fine-tuning  
- Output: Binary classification (Fresh / Spoiled)  
## ⚙️ Usage
```bash
python train.py
python app.py

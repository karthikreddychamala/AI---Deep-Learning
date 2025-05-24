## AI-Powered Pneumonia Detection with Explainable Deep Learning

## 💡 Overview

This project leverages **Convolutional Neural Networks (CNNs)** such as ResNet50, DenseNet121, EfficientNetB0, and MobileNetV2 to classify chest X-rays as either "Normal" or "Pneumonia". Alongside, it integrates **Grad-CAM** for visual explanations, helping understand what the model is actually focusing on.

## 🔧 What It Does

- **Loads a medical image dataset** from Hugging Face (`hf-vision/chest-xray-pneumonia`)
- **Preprocesses data** using PyTorch transforms (including grayscale conversion and normalization)
- **Trains models** with various CNN backbones using transfer learning
- **Evaluates performance** using accuracy, precision, recall, and F1-score
- **Generates visual explanations** using Grad-CAM to highlight areas influencing predictions

## 📦 Requirements

Install the core dependencies using pip:
```bash
pip install tensorflow keras torch torchvision torchaudio opencv-python matplotlib seaborn scikit-learn datasets grad-cam
```

## 🧪 How It Works (In Brief)

1. **Preprocessing**: Chest X-rays are resized, normalized, and augmented.
2. **Model Architecture**: A pre-trained CNN is fine-tuned for binary classification.
3. **Training Loop**: Models are trained and validated with custom loaders and optional class balancing.
4. **Evaluation**: Includes classification reports and confusion matrices.
5. **Grad-CAM**: Visual heatmaps show which parts of an X-ray contributed most to the prediction.

## 🧠 Models Used

You can try out different architectures:
- `resnet18` ✅ (visualized with Grad-CAM)
- `densenet121`
- `mobilenet_v2`
- `efficientnet_b0`
- `vgg16`

The best model is picked based on validation accuracy.

## 📊 Sample Outputs

- Model accuracy printed for each backbone
- Confusion matrix + classification report
- Grad-CAM heatmaps showing:
  - Correct predictions
  - False positives (Normal → Pneumonia)
  - False negatives (Pneumonia → Normal)

## 🖼 Example Visualization (Grad-CAM)

```text
Figure 8: Grad-CAM – Correct Pneumonia Prediction
Figure 9: Grad-CAM – False Positive (Normal → Pneumonia)
Figure 10: Grad-CAM – False Negative (Pneumonia → Normal)
```

## 🗂 File Structure

```text
main.py                # Core script with everything from loading data to training and Grad-CAM
README.md              # You're reading it!
best_model.pth         # Best-performing model checkpoint

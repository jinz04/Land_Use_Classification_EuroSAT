# EuroSAT Satellite Image Classification

A deep learning project for classifying satellite images using ensemble models and advanced training techniques.

## **About**

This project classifies satellite images from the EuroSAT dataset into 10 land cover categories using a sophisticated ensemble of deep learning models.

**Key Results:**
- **Best Accuracy**: 98.30%
- **Final Accuracy**: 96.93%
- **Model**: Ensemble (EfficientNet-B3 + ConvNeXt-Tiny + ResNet50)

## **Dataset**

The EuroSAT dataset contains 27,000 labeled satellite images across 10 land cover classes:

- **AnnualCrop** - Agricultural fields with seasonal crops
- **Forest** - Dense wooded areas
- **HerbaceousVegetation** - Grasslands and meadows
- **Highway** - Major roads and highways
- **Industrial** - Factories and industrial zones
- **Pasture** - Grazing lands for animals
- **PermanentCrop** - Orchards and vineyards
- **Residential** - Housing areas and neighborhoods
- **River** - Water bodies and streams
- **SeaLake** - Oceans, seas, and large lakes

**Dataset Statistics:**
- **Total Images**: 27,000
- **Training Samples**: 21,600 (80%)
- **Validation Samples**: 5,400 (20%)
- **Image Size**: 64×64 pixels (resized to 224×224 for training)
- **Balanced Distribution**: Equal samples per class

## **Exploratory Data Analysis (EDA)**

<img width="1953" height="1190" alt="EDA Analysis" src="https://github.com/user-attachments/assets/a4dd8a36-11d6-468e-a082-e069430b9102" />

*Dataset analysis showing class distribution and sample images*

**Key EDA Insights:**
- Balanced class distribution with 2,700 images per class
- RGB channel analysis shows natural color distributions
- Diverse geographical features across Europe
- High-quality satellite imagery with clear visual distinctions

## **What's Inside**

### **Advanced Features**
- **Ensemble Learning**: Combines multiple models for better accuracy
- **Smart Training**: Label smoothing, CutMix, mixed precision
- **Model Analysis**: Confidence calibration, error analysis
- **Explainable AI**: Understand why the model makes decisions
- **Memory Efficient**: Handles large models without crashing

### **Classes**
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## **Results**

### **Performance Overview**

| Metric | Score |
|--------|-------|
| Best Validation Accuracy | 98.30% |
| Final Validation Accuracy | 96.93% |
| Test Accuracy | 96.93% |
| Macro F1-Score | 96.88% |

### **Top Performing Classes**
- **SeaLake**: 99.4% F1-score
- **Industrial**: 98.8% F1-score
- **Residential**: 98.6% F1-score

<img width="1990" height="1590" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/adbe710b-e994-4e80-83c5-9ec8ef8e5949" />

*Confusion matrix showing model predictions across all classes*

### **Model Calibration**
- **Calibration Error**: 0.0940 (excellent)
- **Model is slightly underconfident**
- Mean Confidence: 87.56% vs Accuracy: 96.93%

<img width="1490" height="990" alt="Training History" src="https://github.com/user-attachments/assets/81e2a2a0-68ac-4b22-a2e4-2c66e9b1980a" />

*Training and validation metrics showing model convergence*

## **Quick Start**

## **Installation**
```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm
pip install torchcam captum
```
## **Basic Usage**

### Train the model
trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.train()

### Evaluate
evaluator = ComprehensiveModelEvaluator(model, val_loader, class_names, device)
results = evaluator.comprehensive_evaluation()

### Explain predictions
xai_analyzer = MultiMethodXAI(model, device)
explanations = xai_analyzer.generate_explanations(image_tensor)

## 🏗️ **Model Architecture**
Ensemble Components:
**EfficientNet-B3** - Efficient and accurate

**ConvNeXt-Tiny** - Modern architecture

**ResNet50** - Proven performer

The ensemble automatically learns which models to trust most using learnable weights.

## ⚙️ **Training Features**
### **Smart Augmentations**
Random flips, rotations, color changes

CutMix: Combins parts of different images

Automatic image adjustments

### **Optimization**
Batch Size: 64

Learning Rate: 0.001 with cosine scheduling

Epochs: 35 (usually converges in 15-20)

Mixed Precision: 2x faster training
<img width="1490" height="990" alt="image" src="https://github.com/user-attachments/assets/81e2a2a0-68ac-4b22-a2e4-2c66e9b1980a" />

*Training and validation metrics showing model convergence*

## 🔍 **Explainable AI**
Justify why the model makes its predictions with multiple explanation methods:

Grad-CAM: Shows which image regions mattered most

Integrated Gradients: Highlights important pixels

Saliency Maps: Gradient-based importance


## 📚 **Citation**
@article{helber2019eurosat,
  title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019}
}

# Plant Disease Classification

A deep learning system for plant leaf disease detection using **transfer learning with EfficientNet-B0** trained on the **PlantVillage dataset**. The project includes a full machine learning pipeline: data preprocessing, model training, evaluation, experiment tracking, and deployment via a **Streamlit web application** for real-time predictions.

---

# System Architecture

```mermaid
flowchart TD
    DS[(PlantVillage Dataset\n38 classes · ~54K images)] --> data

    subgraph data [Data Pipeline — src/data.py]
        direction LR
        tr[Train 80%\nAugmented]
        vl[Val 10%\nCenter-crop]
        ts[Test 10%\nCenter-crop]
    end

    subgraph model [Model Architecture — src/model.py]
        direction TB
        backbone["EfficientNet-B0\n(ImageNet pretrained)"]
        head["Custom Head\nLinear(1280 → 38 classes)"]
        backbone --> head
    end

    data --> model

    subgraph training [Training Pipeline — src/train.py]
        p1["Phase 1 — Head Only\nfreeze backbone · Adam lr=1e-3 · 2 epochs"]
        p2["Phase 2 — Fine-Tune\nunfreeze last 2 blocks · Adam lr=1e-4 · 5 epochs"]
        p1 --> p2
    end

    tr --> training
    vl --> training
    model --> training

    training -->|Best val Macro F1| ckpt[(artifacts/models/\nmodel_vN.pt)]
    training -->|Metrics and curves| wb[Weights and Biases\nExperiment Tracking]

    ckpt --> eval
    ckpt --> serve

    subgraph eval [Evaluation — src/eval.py]
        direction LR
        ts2[Test Set 10%] --> scores["Accuracy  98.21%\nMacro F1  0.9765\nWeighted F1  0.9821"]
        scores --> reports["artifacts/reports/\nmetrics.json\nclassification_report.txt\nconfusion_matrix.png"]
    end

    subgraph serve [Inference and Deployment]
        direction TB
        cli["CLI — src/predict.py\nTop-K predictions from image path"]
        app["Streamlit App — app/streamlit_app.py\nUpload image → Predict → Display results\nLogs to logs/predictions.log"]
    end
```

---

# Model Architecture

The project uses **EfficientNet-B0** pretrained on ImageNet.

Training follows a two-phase transfer learning strategy:

### Phase 1 — Train Classification Head
- Freeze the EfficientNet backbone
- Train the classifier layer

### Phase 2 — Fine-Tune Backbone
- Unfreeze the last layers of EfficientNet
- Fine-tune with a smaller learning rate

This improves generalization while reducing training time.

---

# Model Performance

Evaluation on the held-out test set:

| Metric | Score |
|------|------|
| Accuracy | **98.21%** |
| Macro F1 Score | **0.9765** |
| Weighted F1 Score | **0.9821** |

---

Dataset source:

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

---

# Tech Stack

### Machine Learning
- PyTorch
- Torchvision
- EfficientNet-B0 (transfer learning)

### Data Processing
- OpenCV
- PIL
- NumPy

### Evaluation
- Matplotlib

### Experiment Tracking
- Weights & Biases (W&B)

### Deployment
- Streamlit

### Development
- Python
- Git / GitHub
- VS Code
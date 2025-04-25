```markdown
# FairBias

A comprehensive framework for multi-domain medical image analysis, featuring co-training, domain adaptation, and quality-aware semi-supervised learning. Designed for datasets like CheXpert, Breast MRI/FFDM, Hep-2, and SOKL.

---

## 📂 Datasets

### Overview
The framework supports the following datasets:

| Dataset Name         | Description                                                                 | Modality          | Size (Images) | Classes | Source                                                                 |
|----------------------|-----------------------------------------------------------------------------|-------------------|---------------|---------|------------------------------------------------------------------------|
| **CheXpert**         | Chest radiographs with uncertainty labels for 14 pathologies.               | X-ray             | ~224,316      | 14      | [arXiv:1901.07031](https://arxiv.org/abs/1901.07031)                  |
| **Breast MRI/FFDM**  | Breast cancer imaging (MRI and Full-Field Digital Mammography).             | MRI/FFDM          | Varies        | 2-4     | [TCIA](https://www.cancerimagingarchive.net/)                         |
| **Hep-2**            | Human epithelial (HEp-2) cell images for autoimmune disease diagnosis.      | Microscopy        | ~11,000       | 6       | [Kaggle](https://www.kaggle.com/datasets/arunava007/hep2-cell-images) |
| **SOKL**             | Kidney pathology images for structural and functional analysis.             | Ultrasound/CT/MRI | ~5,000        | 5       | [KidneyImage.org](https://www.kidneyimage.org/)                       |

### Configuration
Update `DATASET_PATHS` in `MedicalDataset.py` with your local paths:
```python
DATASET_PATHS = {
    "CheXpert": {"images_dir": "your/path", "labels_file": "your/path"},
    # Repeat for other datasets
}
```

---

## 🛠️ Code Structure

### Key Components

#### 1. `ClassAwareGMM.py`
- **Purpose**: Class-aware Gaussian Mixture Model for noise sample separation.
- **Classes**:
  - `ClassAwareGMM`: Fits GMMs to loss/uncertainty features per class.
  - **Methods**: `fit()`, `predict()`.

#### 2. `CoTraining.py`
- **Purpose**: Co-training scheme with two networks and SSL.
- **Classes**:
  - `CoTraining`: Alternates training between two models.
  - `SampleReweighingLoss`: Adjusts weights for noisy samples.
  - `ContrastiveEnhancementLoss`: Enhances feature separation.

#### 3. `MedicalDataset.py`
- **Purpose**: Dataset loader for medical images.
- **Classes**:
  - `MedicalDataset`: Loads images/labels from CSV and applies transformations.

#### 4. `MultiDomainAdaptation.py`
- **Purpose**: Aligns features across domains (e.g., CheXpert → SOKL).
- **Classes**:
  - `MultiDomainAdaptation`: Implements domain-invariant training, contrastive learning, and fine-tuning.
  - `DomainSpecificBatchNorm`: Domain-specific normalization.

#### 5. `QualityAwareSSLTraining.py`
- **Purpose**: Semi-supervised training with noise-aware losses.
- **Classes**:
  - `EnhancedSampleSeparation`: Identifies clean/noisy samples using GMM.
  - `QualityAwareSSLTraining`: Combines CLS, MHQS, and MLQS losses.

---

## 🚀 Usage

### Installation
```bash
pip install torch torchvision scikit-learn umap-learn
```

### Training Pipeline
1. **Prepare Datasets**: Configure paths in `MedicalDataset.py`.
2. **Run Main Pipeline**:
```python
python main.py
```
3. **Custom Training** (Example):
```python
from CoTraining import CoTraining
from MultiDomainAdaptation import MultiDomainAdaptation

# Initialize models and components
mda = MultiDomainAdaptation(...)
co_trainer = CoTraining(model_a, model_b, num_classes=5)
co_trainer.train(dataloader, optimizer_a, optimizer_b, total_epochs=100)
```

---

## 📊 Visualization
UMAP embeddings are generated for feature visualization:
```python
visualize_umap(embeddings, labels, title="Feature Embeddings")
```
![UMAP Example](https://via.placeholder.com/600x400?text=UMAP+Visualization)

---

## 📚 References
- CheXpert: [Irvin et al. (2019)](https://arxiv.org/abs/1901.07031)
- Breast MRI/FFDM: [TCIA](https://www.cancerimagingarchive.net/)
- Hep-2: [Kaggle](https://www.kaggle.com/datasets/arunava007/hep2-cell-images)
- SOKL: [KidneyImage.org](https://www.kidneyimage.org/)
```

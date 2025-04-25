import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --------------------------- Dataset Handling ----------------------------
class MedicalDataset(Dataset):
    def __init__(self, dataset_name, images, labels, transform=None):
        """
        Initialize a medical dataset.
        :param dataset_name: Name of the dataset (e.g., "CheXpert", "Breast_MRI_FFD").
        :param images: List of image paths or tensors.
        :param labels: List of one-hot encoded labels.
        :param transform: Optional transformations to apply to images.
        """
        self.dataset_name = dataset_name
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, str):  # If image is a file path
            image = Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------- Enhanced Sample Separation ------------------
class EnhancedSampleSeparation:
    def __init__(self, num_classes, class_weights=None):
        """
        Initialize enhanced sample separation using a Gaussian Mixture Model (GMM).
        :param num_classes: Total number of classes in the dataset.
        :param class_weights: Class-balancing weights to account for class imbalance.
        """
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.gmms = {}

    def fit(self, losses, uncertainties, labels):
        """
        Fit a class-balanced GMM to model the joint distribution of normalized loss and uncertainty.
        :param losses: Array of computed losses for all samples.
        :param uncertainties: Array of computed uncertainties for all samples.
        :param labels: Array of one-hot encoded labels.
        """
        unique_classes = np.unique(labels)
        for c in unique_classes:
            class_mask = labels == c
            class_losses = losses[class_mask]
            class_uncertainties = uncertainties[class_mask]

            # Normalize loss and uncertainty
            normalized_losses = (class_losses - np.min(class_losses)) / (
                np.max(class_losses) - np.min(class_losses)
            )
            normalized_uncertainties = (class_uncertainties - np.min(class_uncertainties)) / (
                np.max(class_uncertainties) - np.min(class_uncertainties)
            )

            features = np.column_stack((normalized_losses, normalized_uncertainties))
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(features)
            self.gmms[c] = gmm

    def predict(self, losses, uncertainties, labels):
        """
        Predict noise indicators for each sample based on the fitted GMMs.
        :param losses: Array of computed losses for all samples.
        :param uncertainties: Array of computed uncertainties for all samples.
        :param labels: Array of one-hot encoded labels.
        :return: Noise indicators (0: CLS, 1: MHQS, 2: MLQS).
        """
        unique_classes = np.unique(labels)
        predictions = np.zeros_like(labels)

        for c in unique_classes:
            class_mask = labels == c
            class_losses = losses[class_mask]
            class_uncertainties = uncertainties[class_mask]

            # Normalize loss and uncertainty
            normalized_losses = (class_losses - np.min(class_losses)) / (
                np.max(class_losses) - np.min(class_losses)
            )
            normalized_uncertainties = (class_uncertainties - np.min(class_uncertainties)) / (
                np.max(class_uncertainties) - np.min(class_uncertainties)
            )

            features = np.column_stack((normalized_losses, normalized_uncertainties))
            posterior_probs = self.gmms[c].predict_proba(features)
            predictions[class_mask] = np.argmax(posterior_probs, axis=1)

        return predictions


# --------------------------- Quality-Aware Semi-Supervised Training ------
class QualityAwareSSLTraining:
    def __init__(self, model, num_classes, lambda_cls=0.5, beta_mlqs=0.1, tau_contrastive=0.07):
        """
        Initialize quality-aware semi-supervised training.
        :param model: Neural network model for disease diagnosis.
        :param num_classes: Total number of classes.
        :param lambda_cls: Weight for dynamic class reweighting.
        :param beta_mlqs: Decay factor for MLQS sample reweighing loss.
        :param tau_contrastive: Temperature parameter for contrastive learning.
        """
        self.model = model
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.beta_mlqs = beta_mlqs
        self.tau_contrastive = tau_contrastive

    def compute_loss(self, outputs, targets, noise_indicators, epoch, total_epochs):
        """
        Compute the combined loss for quality-aware SSL training.
        :param outputs: Model predictions.
        :param targets: Ground truth labels.
        :param noise_indicators: Noise indicators (0: CLS, 1: MHQS, 2: MLQS).
        :param epoch: Current training epoch.
        :param total_epochs: Total number of epochs.
        :return: Combined loss value.
        """
        cls_mask = noise_indicators == 0
        mlqs_mask = noise_indicators == 2

        # Dynamic class reweighting for CLS
        cls_loss = -torch.mean(
            self.lambda_cls * torch.sum(targets[cls_mask] * torch.log(outputs[cls_mask]), dim=1)
        )

        # Sample-reweighing loss for MLQS with decaying weight
        omega_t = max(0, 1 - epoch / total_epochs)
        mlqs_loss = -omega_t * torch.mean(
            self.beta_mlqs * torch.sum(targets[mlqs_mask] * torch.log(outputs[mlqs_mask]), dim=1)
        )

        # Contrastive learning loss
        embeddings = self.model.encoder(outputs)
        sim_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        exp_sim = torch.exp(sim_matrix / self.tau_contrastive)
        contrastive_loss = -torch.mean(
            torch.log(exp_sim.diag() / torch.sum(exp_sim, dim=1))
        )

        # Combined loss
        total_loss = cls_loss + mlqs_loss + contrastive_loss
        return total_loss


# --------------------------- Multi-Domain Adaptation ---------------------
class MultiDomainAdaptation:
    def __init__(self, feature_extractor, domain_discriminator, lambda_domain=0.5, beta_finetune=0.1, tau_contrastive=0.07):
        """
        Initialize multi-domain adaptation components.
        :param feature_extractor: Neural network for extracting domain-invariant features.
        :param domain_discriminator: Neural network for distinguishing between domains.
        :param lambda_domain: Trade-off weight for domain alignment.
        :param beta_finetune: Regularization weight for fine-tuning.
        :param tau_contrastive: Temperature parameter for cross-domain contrastive learning.
        """
        self.feature_extractor = feature_extractor
        self.domain_discriminator = domain_discriminator
        self.lambda_domain = lambda_domain
        self.beta_finetune = beta_finetune
        self.tau_contrastive = tau_contrastive

    def align_domains(self, source_features, target_features):
        """
        Align representations across domains using adversarial training.
        :param source_features: Features from the source domain.
        :param target_features: Features from the target domain.
        :return: Domain alignment loss.
        """
        domain_labels_source = torch.zeros(source_features.size(0), dtype=torch.long)
        domain_labels_target = torch.ones(target_features.size(0), dtype=torch.long)
        domain_labels = torch.cat([domain_labels_source, domain_labels_target])

        combined_features = torch.cat([source_features, target_features])
        domain_predictions = self.domain_discriminator(combined_features)
        domain_loss = torch.nn.CrossEntropyLoss()(domain_predictions, domain_labels)

        return domain_loss

    def fine_tune(self, target_features, target_labels):
        """
        Fine-tune the model on the target domain with regularization.
        :param target_features: Features from the target domain.
        :param target_labels: Labels from the target domain.
        :return: Fine-tuning loss.
        """
        task_loss = torch.nn.CrossEntropyLoss()(target_features, target_labels)
        regularization_loss = self.beta_finetune * torch.norm(target_features, p=2)
        return task_loss + regularization_loss

    def cross_domain_contrastive_learning(self, source_embeddings, target_embeddings):
        """
        Perform cross-domain contrastive learning.
        :param source_embeddings: Embeddings from the source domain.
        :param target_embeddings: Embeddings from the target domain.
        :return: Cross-domain contrastive loss.
        """
        sim_matrix = torch.cosine_similarity(source_embeddings.unsqueeze(1), target_embeddings.unsqueeze(0), dim=-1)
        exp_sim = torch.exp(sim_matrix / self.tau_contrastive)
        contrastive_loss = -torch.mean(
            torch.log(exp_sim.diag() / torch.sum(exp_sim, dim=1))
        )
        return contrastive_loss


# --------------------------- Main Function -------------------------------
def main():
    # Dummy paths for datasets
    DATASET_PATHS = {
        "CheXpert": {"images_dir": "/data/CheXpert/images/", "labels_file": "/data/CheXpert/labels.csv"},
        "Breast_MRI_FFD": {"images_dir": "/data/Breast_MRI_FFD/images/", "labels_file": "/data/Breast_MRI_FFD/labels.csv"},
        "Hep2": {"images_dir": "/data/Hep2/images/", "labels_file": "/data/Hep2/labels.csv"},
        "SOKL": {"images_dir": "/data/SOKL/images/", "labels_file": "/data/SOKL/labels.csv"},
    }

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Initialize datasets
    medical_datasets = {}
    for name, data in DATASET_PATHS.items():
        medical_datasets[name] = MedicalDataset(name, data["images"], data["labels"], transform=transform)

    # Define dataloaders
    dataloaders = {name: DataLoader(ds, batch_size=16, shuffle=True) for name, ds in medical_datasets.items()}

    # Initialize models and components
    feature_extractor = FeatureExtractor(pretrained=True)
    domain_discriminator = DomainDiscriminator()
    model = DiseaseDiagnosisModel(num_classes=5)

    # Initialize integrated methodology components
    sample_separation = EnhancedSampleSeparation(num_classes=5)
    ssl_training = QualityAwareSSLTraining(model, num_classes=5)
    domain_adaptation = MultiDomainAdaptation(feature_extractor, domain_discriminator)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    total_epochs = 100
    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        for name, dataloader in dataloaders.items():
            for images, labels in dataloader:
                # Step 3.1: Enhanced Sample Separation
                outputs = model(images)
                losses = -torch.sum(labels * torch.log(outputs + 1e-10), dim=1)
                uncertainties = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)
                noise_indicators = sample_separation.predict(losses.detach().numpy(), uncertainties.detach().numpy(), labels.numpy())

                # Step 3.2: Quality-Aware Semi-Supervised Training
                loss = ssl_training.compute_loss(outputs, labels, noise_indicators, epoch, total_epochs=total_epochs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step 3.3: Multi-Domain Adaptation
                source_features = feature_extractor(images)  # Source domain features
                target_features = feature_extractor(images)  # Replace with target domain data if available

                # Align domains using adversarial training
                domain_loss = domain_adaptation.align_domains(source_features, target_features)

                # Fine-tune on target domain with regularization
                fine_tune_loss = domain_adaptation.fine_tune(target_features, labels)

                # Cross-domain contrastive learning
                contrastive_loss = domain_adaptation.cross_domain_contrastive_learning(source_features, target_features)

                # Combine domain adaptation losses
                domain_adaptation_loss = domain_loss + fine_tune_loss + contrastive_loss

                # Backpropagation for domain adaptation
                optimizer.zero_grad()
                domain_adaptation_loss.backward()
                optimizer.step()

        # Optional: Log metrics or visualize embeddings
        if (epoch + 1) % 10 == 0:
            print("Logging metrics and visualizing embeddings...")
            with torch.no_grad():
                embeddings = feature_extractor(images)
                visualize_umap(embeddings, labels, title=f"UMAP Visualization at Epoch {epoch + 1}")


# --------------------------- Utility Functions ---------------------------
def visualize_umap(embeddings, labels, title="UMAP Visualization"):
    """
    Visualize embeddings using UMAP.
    :param embeddings: Tensor of embeddings to visualize.
    :param labels: Ground truth labels for coloring.
    :param title: Title of the plot.
    """
    import umap
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels.cpu().numpy(), cmap='tab10', s=5)
    plt.title(title)
    plt.colorbar(label="Class Labels")
    plt.show()
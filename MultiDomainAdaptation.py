import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np


# --------------------------- Feature Extractor ---------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.backbone(x)


# --------------------------- Domain Discriminator ------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=2048):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # Binary output: source vs target

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


# --------------------------- Projection Head -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# --------------------------- Domain-Specific Batch Normalization --------
class DomainSpecificBatchNorm(nn.Module):
    def __init__(self, num_features, num_domains):
        super(DomainSpecificBatchNorm, self).__init__()
        self.num_domains = num_domains
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_domains)])

    def forward(self, x, domain_idx):
        return self.bns[domain_idx](x)


# --------------------------- Multi-Domain Adaptation ---------------------
class MultiDomainAdaptation:
    def __init__(self, source_datasets, target_dataset, num_classes, lambda_=0.5, beta=0.1, tau=0.07, num_domains=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_ = lambda_
        self.beta = beta
        self.tau = tau
        self.num_domains = num_domains

        # Initialize models
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.domain_discriminator = DomainDiscriminator().to(self.device)
        self.projection_head = ProjectionHead().to(self.device)
        self.classifier = nn.Linear(2048, num_classes).to(self.device)
        self.domain_bn = DomainSpecificBatchNorm(2048, num_domains).to(self.device)

        # Optimizers
        self.optimizer_F = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        self.optimizer_D = optim.Adam(self.domain_discriminator.parameters(), lr=1e-4)
        self.optimizer_C = optim.Adam(self.classifier.parameters(), lr=1e-4)

    def train_domain_invariance(self, source_loader, target_loader):
        """
        Train the feature extractor to minimize domain discrepancy.
        """
        self.feature_extractor.train()
        self.domain_discriminator.train()

        for (x_s, _), (x_t, _) in zip(source_loader, target_loader):
            x_s, x_t = x_s.to(self.device), x_t.to(self.device)

            # Forward pass through feature extractor
            f_s = self.feature_extractor(x_s)
            f_t = self.feature_extractor(x_t)

            # Domain discriminator loss
            d_s = self.domain_discriminator(f_s)
            d_t = self.domain_discriminator(f_t)
            L_D = -torch.mean(torch.log(d_s)) - torch.mean(torch.log(1 - d_t))

            # Feature extractor loss
            L_F = L_D + self.lambda_ * torch.mean(torch.abs(f_s - f_t))

            # Update domain discriminator
            self.optimizer_D.zero_grad()
            L_D.backward(retain_graph=True)
            self.optimizer_D.step()

            # Update feature extractor
            self.optimizer_F.zero_grad()
            L_F.backward()
            self.optimizer_F.step()

    def fine_tune_transfer_learning(self, target_loader):
        """
        Fine-tune the model on the target domain with regularization.
        """
        self.feature_extractor.train()
        self.classifier.train()

        for x_t, y_t in target_loader:
            x_t, y_t = x_t.to(self.device), y_t.to(self.device)

            # Forward pass
            f_t = self.feature_extractor(x_t)
            logits = self.classifier(f_t)

            # Task-specific loss
            L_task = nn.CrossEntropyLoss()(logits, torch.argmax(y_t, dim=1))

            # Regularization loss
            L_reg = sum(p.norm(2) for p in self.feature_extractor.parameters())

            # Total loss
            L_transfer = L_task + self.beta * L_reg

            # Backward pass
            self.optimizer_C.zero_grad()
            self.optimizer_F.zero_grad()
            L_transfer.backward()
            self.optimizer_C.step()
            self.optimizer_F.step()

    def cross_domain_contrastive_learning(self, source_loader, target_loader):
        """
        Align representations across domains using contrastive learning.
        """
        self.feature_extractor.train()
        self.projection_head.train()

        for (x_s, _), (x_t, _) in zip(source_loader, target_loader):
            x_s, x_t = x_s.to(self.device), x_t.to(self.device)

            # Forward pass through feature extractor and projection head
            f_s = self.feature_extractor(x_s)
            f_t = self.feature_extractor(x_t)
            z_s = self.projection_head(f_s)
            z_t = self.projection_head(f_t)

            # Compute cosine similarity
            sim_st = torch.cosine_similarity(z_s, z_t, dim=1)
            negative_sim = torch.matmul(z_s, z_t.T)

            # Contrastive loss
            numerator = torch.exp(sim_st / self.tau)
            denominator = torch.sum(torch.exp(negative_sim / self.tau), dim=1)
            L_contrastive = -torch.mean(torch.log(numerator / denominator))

            # Backward pass
            self.optimizer_F.zero_grad()
            L_contrastive.backward()
            self.optimizer_F.step()



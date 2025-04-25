# --------------------------- Main Function -------------------------------
def main():
    # Load datasets
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Initialize datasets
    medical_datasets = {}
    for name in DATASET_PATHS.keys():
        medical_datasets[name] = MedicalDataset(name, transform=transform)

    # Define dataloaders
    dataloaders = {
        name: DataLoader(ds, batch_size=16, shuffle=True)
        for name, ds in medical_datasets.items()
    }
    # Initialize multi-domain adaptation framework
    mda = MultiDomainAdaptation(
        source_datasets=["CheXpert", "Breast_MRI_FFD", "Hep2"],
        target_dataset="SOKL",
        num_classes=5,
        lambda_=0.5,
        beta=0.1,
        tau=0.07,
        num_domains=4,
    )

    # Training loop
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        mda.train_domain_invariance(dataloaders["CheXpert"], dataloaders["SOKL"])
        mda.fine_tune_transfer_learning(dataloaders["SOKL"])
        mda.cross_domain_contrastive_learning(dataloaders["CheXpert"], dataloaders["SOKL"])


if __name__ == "__main__":
    main()
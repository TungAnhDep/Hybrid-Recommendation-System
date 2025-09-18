import torch
from torch.utils.data import Dataset, DataLoader


class RatingDataset(Dataset):
    def __init__(self, df, features, context):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.products = torch.tensor(df["product"].values, dtype=torch.long)
        self.features = torch.tensor(df[features].values, dtype=torch.float32)
        self.context = torch.tensor(df[context].values, dtype=torch.float32)
        self.labels = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.products[idx],
            self.features[idx],
            self.context[idx],
            self.labels[idx],
        )


def create_dataloader(train_df, test_df, features, context, batch_size=64, num_workers=0):
    train_dataset = RatingDataset(train_df, features, context)
    test_dataset = RatingDataset(test_df, features, context)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

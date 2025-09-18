import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.models.ncf import HybridRecommender
from dataloader import create_dataloader
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor
from src.data.split import DataSplitter
from helper.utils import load_recommender_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for users, products, features, contexts, labels in tqdm(dataloader, desc="Training"):
        users, products = users.to(device), products.to(device)
        features, contexts, labels = features.to(device), contexts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(users, products, features, contexts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * users.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for users, products, features, contexts, labels in tqdm(dataloader, desc="Validating"):
            users, products = users.to(device), products.to(device)
            features, contexts, labels = features.to(device), contexts.to(device), labels.to(device)

            outputs = model(users, products, features, contexts)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * users.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


if __name__ == "__main__":
    final_products, users, context, device, model, features, user_item_matrix = load_recommender_model(
    model_path="hybrid_model.pth"
)
    # Split & build train/test
    splitter = DataSplitter()
    dtf_train, dtf_test, _ = splitter.split_and_scale(user_item_matrix)
    train_df, test_df, features, context_cols = splitter.build_train_test(dtf_train, dtf_test, final_products, context)
    
    # Dataloader
    train_loader, test_loader = create_dataloader(train_df, test_df, features, context_cols)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRecommender(
        usr=len(users["user"].unique()),
        prd=len(final_products),
        feat=len(features),
        ctx=len(context_cols),
        embedding_size=32
    ).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(3):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "hybrid_model.pth")

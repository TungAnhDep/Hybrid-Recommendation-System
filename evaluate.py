import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from dataloader import create_dataloader
from src.models.ncf import HybridRecommender
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor
from src.data.split import DataSplitter
from helper.utils import load_recommender_model

def evaluate(model, dataloader, criterion, device, splitter):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for users, products, features, contexts, labels in tqdm(dataloader, desc="Evaluating"):
            users, products = users.to(device), products.to(device)
            features, contexts, labels = features.to(device), contexts.to(device), labels.to(device)

            outputs = model(users, products, features, contexts)

            # Loss trên thang đo đã scale
            loss = criterion(outputs, labels)
            running_loss += loss.item() * users.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # numpy
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)

    # === Inverse Rating===
    inv_labels = splitter.inverse_ratings(all_labels)
    inv_preds  = splitter.inverse_ratings(all_preds)

    
    rmse = (root_mean_squared_error(inv_labels, inv_preds))
    mae  = mean_absolute_error(inv_labels, inv_preds)

    return epoch_loss, rmse, mae


if __name__ == "__main__":
    final_products, users, context, device, model, features, user_item_matrix = load_recommender_model(
    model_path="hybrid_model.pth"
)
    # Split & build train/test
    splitter = DataSplitter()
    dtf_train, dtf_test, _ = splitter.split_and_scale(user_item_matrix)
    train_df, test_df, features, context_cols = splitter.build_train_test(dtf_train, dtf_test, final_products, context)

    # Dataloader
    _, test_loader = create_dataloader(train_df, test_df, features, context_cols, batch_size=128)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRecommender(
        usr=len(users["user"].unique()),
        prd=len(final_products),
        feat=len(features),
        ctx=len(context_cols),
        embedding_size=32
    ).to(device)

    model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))
    criterion = nn.MSELoss()

    # Evaluate
    val_loss, rmse, mae = evaluate(model, test_loader, criterion, device, splitter)
    print(f"RMSE (original scale) = {rmse:.4f}")
    print(f"MAE  (original scale) = {mae:.4f}")

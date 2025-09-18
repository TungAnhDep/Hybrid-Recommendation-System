import torch
import numpy as np
import pandas as pd
from src.models.ncf import HybridRecommender
from src.data.split import DataSplitter
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor
from dataloader import create_dataloader
from  helper.utils import load_recommender_model


def predict_to_dataframe(model, df, features, context, product, device="cpu"):
    model.eval()

    # Convert to tensor
    users = torch.tensor(df["user"].values, dtype=torch.long, device=device)
    products = torch.tensor(df["product"].values, dtype=torch.long, device=device)
    feat = torch.tensor(df[features].values, dtype=torch.float32, device=device)
    ctx = torch.tensor(df[context].values, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(users, products, feat, ctx)  # (batch, 1)

    # Chuyển tensor -> pandas.Series rồi gắn vào DataFrame
    df = df.copy()
    df["yhat"] = pd.Series(outputs.squeeze(1).cpu().numpy(), index=df.index)
    df = df.merge(product['name'],  how = "left", left_on = 'product', right_index = True)    
    return df
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
    test_df_pred = predict_to_dataframe(model, test_df, features, context_cols, final_products,device=device)
    print(test_df_pred.head())


    








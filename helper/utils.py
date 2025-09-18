import torch
from src.models.ncf import HybridRecommender
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor

def load_recommender_model(model_path="hybrid_model.pth", embedding_size=32, ctx_size=2, device=None):

    # 1. Load dataset
    dataset = MovieLensDataset()
    movies, ratings = dataset.load()
    
    # 2. Preprocess
    products = Preprocessor.process_products(movies)
    users, context = Preprocessor.process_users(ratings, products)
    final_products = Preprocessor.encode_genres(products)
    user_item_matrix = Preprocessor.build_user_item_matrix(users, final_products)
    # 3. Features list
    features = [col for col in final_products.columns if col not in ["name", "old", "product"]]
    
    # 4. Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5. Load model
    model = HybridRecommender(
        usr=len(users["user"].unique()),
        prd=len(products),
        feat=len(features),
        ctx=ctx_size,
        embedding_size=embedding_size
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return final_products, users, context, device, model, features, user_item_matrix

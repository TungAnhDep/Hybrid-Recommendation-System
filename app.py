import streamlit as st
import torch
import numpy as np
import pandas as pd
from src.models.ncf import HybridRecommender
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor
from helper.utils import load_recommender_model
def recommend_movies(model, user_id, context_row, movies_df, features, device="cpu", top_k=10):
    model.eval()

    users = torch.tensor([user_id] * len(movies_df), dtype=torch.long, device=device)
    products = torch.arange(len(movies_df), dtype=torch.long, device=device)
    feat = torch.tensor(movies_df[features].values, dtype=torch.float32, device=device)
    ctx = torch.tensor(np.repeat([context_row], len(movies_df), axis=0),
                       dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(users, products, feat, ctx)

    movies_df = movies_df.copy()
    movies_df["yhat"] = outputs.squeeze(1).cpu().numpy()

    genre_cols = [c for c in features]
    def join_genres(row):
        genres = [g for g in genre_cols if row[g] == 1]
        return "|".join(genres) if genres else "No-Genre"

    movies_df["Genres"] = movies_df.apply(join_genres, axis=1)
    return movies_df.sort_values("yhat", ascending=False).head(top_k)[["name", "Genres"]]

if __name__ == "__main__":
    st.title("🎬 Movie Recommender Demo")
    final_products, users, context, device, model, features, _ = load_recommender_model(
    model_path="hybrid_model.pth"
)

    model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))

    st.header("🔮 Recommend Movies")

    with st.form("recommend_form"):
        user_id = st.number_input(
            "Enter your ID", 
            min_value=int(users["user"].min()), 
            max_value=int(users["user"].max()), 
            value=int(users["user"].min()), 
            step=1
        )
        N = st.slider("Number of recommended movies", 1, 20, 5)

        # Context
        daytime = st.selectbox("Time", ["Day", "Night"])
        weekend = st.checkbox("Weekends?")

        selected_genres = st.multiselect("Genres", features, default=[])

        recommend_btn = st.form_submit_button("Recommend")

    if recommend_btn:
        context_row = np.array([
            1 if daytime == "Ban ngày (6-20h)" else 0,
            1 if weekend else 0
        ], dtype=np.float32)

        movies_to_use = final_products.copy()

        if selected_genres:
            mask = movies_to_use[selected_genres].sum(axis=1) > 0
            movies_to_use = movies_to_use[mask]
        top_movies = recommend_movies(
                model, int(user_id), context_row, movies_to_use, features,
                device=device, top_k=N
        )

        st.subheader(f"Top {N} phim gợi ý cho user {user_id}")
        pd.set_option('display.max_colwidth', None) 
        st.table(top_movies)

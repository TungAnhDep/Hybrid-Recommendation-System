from src.data.dataset import MovieLensDataset
import re
import numpy as np
import pandas as pd
from datetime import datetime


class Preprocessor:
    """
    Xử lý dữ liệu MovieLens:
    - Làm sạch thông tin sản phẩm (movies)
    - Xử lý thông tin người dùng (ratings)
    - Trích xuất context features
    - One-hot encode Genres
    - Pivot user-item matrix
    """

    @staticmethod
    def process_products(dtf_products: pd.DataFrame) -> pd.DataFrame:
        """Xử lý thông tin phim (products)"""
        # Bỏ các record thiếu Genres
        dtf_products = dtf_products[~dtf_products["Genres"].isna()].copy()

        # Gán ID sản phẩm (0...N-1)
        dtf_products["product"] = range(len(dtf_products))

        # Làm sạch tên phim (bỏ năm hoặc text trong ngoặc)
        dtf_products["name"] = dtf_products["Title"].apply(
            lambda x: re.sub(r"[\(\[].*?[\)\]]", "", x).strip()
        )

        # Lấy năm phát hành
        def extract_year(x):
            if "(" in x:
                try:
                    return int(x.split("(")[-1].replace(")", "").strip())
                except:
                    return np.nan
            return np.nan

        dtf_products["date"] = dtf_products["Title"].apply(extract_year)
        dtf_products["date"] = dtf_products["date"].fillna(9999)

        # Flag phim cũ (trước 2000)
        dtf_products["old"] = dtf_products["date"].apply(lambda x: 1 if x < 2000 else 0)

        # Giữ lại cột cần thiết
        #dtf_products = dtf_products[["product", "name", "old", "Genres"]].set_index("product")

        return dtf_products

    @staticmethod
    def process_users(dtf_users: pd.DataFrame, dtf_products: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Xử lý thông tin user + context"""
        dtf_users["user"] = dtf_users["UserID"].apply(lambda x: x - 1)

        dtf_users["Timestamp"] = dtf_users["Timestamp"].apply(lambda x: datetime.fromtimestamp(x))

        # Context features
        dtf_users["daytime"] = dtf_users["Timestamp"].apply(lambda x: 1 if 6 < int(x.strftime("%H")) < 20 else 0)
        dtf_users["weekend"] = dtf_users["Timestamp"].apply(lambda x: 1 if x.weekday() in [5, 6] else 0)

        # Join with product ID
        dtf_users = dtf_users.merge(dtf_products[["MovieID", "product"]], how="left")
        

        
        dtf_users = dtf_users.rename(columns={"Rating": "y"})

        dtf_context = dtf_users[["user", "product", "daytime", "weekend"]]
        dtf_users = dtf_users[["user", "product", "y"]]

        return dtf_users, dtf_context

    @staticmethod
    def encode_genres(dtf_products: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode genres"""
        tags = [i.split("|") for i in dtf_products["Genres"].unique()]
        columns = sorted(set([i for lst in tags for i in lst]))

        for col in columns:
            dtf_products[col] = dtf_products["Genres"].apply(lambda x: 1 if col in x else 0)

        return dtf_products[['name', 'old']+columns]

    @staticmethod
    def build_user_item_matrix(dtf_users: pd.DataFrame, dtf_products: pd.DataFrame) -> pd.DataFrame:
        """Pivot user-item matrix"""
        tmp = dtf_users.copy()
        dtf_users_pivot = tmp.pivot_table(index="user", columns="product", values="y")

        missing_cols = list(set(dtf_products.index) - set(dtf_users_pivot.columns))
        missing_df = pd.DataFrame(np.nan, index=dtf_users_pivot.index, columns=missing_cols)

        dtf_users_pivot = pd.concat([dtf_users_pivot, missing_df], axis=1)
        dtf_users_pivot.columns.name = "product"

        dtf_users_pivot = dtf_users_pivot[sorted(dtf_users_pivot.columns)]

        return dtf_users_pivot



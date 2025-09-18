import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.data.dataset import MovieLensDataset
from src.data.preprocess import Preprocessor
class DataSplitter:

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.min_rating = None
        self.max_rating = None

    def split_and_scale(self, ratings: pd.DataFrame):
        train, test = train_test_split(
            ratings, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

        self.min_rating = train.min().min()
        self.max_rating = train.max().max()

        def scale(df):
            return (df - self.min_rating) / (self.max_rating - self.min_rating)

        return scale(train), scale(test), None

    def inverse_ratings(self, y_scaled):
        """Inverse từ [0,1] về thang đo gốc"""
        y_scaled = np.array(y_scaled).astype(float)
        return y_scaled * (self.max_rating - self.min_rating) + self.min_rating
    def build_train_test(self, dtf_train, dtf_test, dtf_products, dtf_context):
        
        features = dtf_products.drop(["name", 'old'], axis=1).columns
        context = dtf_context.drop(["user", "product"], axis=1).columns

        train = dtf_train.stack(dropna=True).reset_index().rename(columns={0: "y"})
        train = train.merge(dtf_products[features], how="left", left_on="product", right_index=True)
        train = train.merge(dtf_context, how="left")

        test = dtf_test.stack(dropna=True).reset_index().rename(columns={0: "y"})
        test = test.merge(dtf_products[features], how="left", left_on="product", right_index=True)
        test = test.merge(dtf_context, how="left")

        return train, test, features, context
    

    


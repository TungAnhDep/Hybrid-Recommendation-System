import pandas as pd

class MovieLensDataset:
    def __init__(self, movies_path="data/raw/movies.dat", ratings_path="data/raw/ratings.dat"):
        self.movies_path = movies_path
        self.ratings_path = ratings_path

    def load(self):
        movies = pd.read_csv(
            self.movies_path,
            sep="::", engine="python", header=None,
            names=["MovieID", "Title", "Genres"],
            encoding="ISO-8859-1"
        )
        ratings = pd.read_csv(
            self.ratings_path,
            sep="::", engine="python", header=None,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            encoding="ISO-8859-1"
        )
        return movies, ratings


# Hybrid Movie Recommendation System

A **movie recommendation system** using Neural Collaborative Filtering + Content-based features + Knowledge-based.  
Built with PyTorch and Streamlit.

---

## 🔹 Features

- Hybrid recommendation: combines **user-item interactions** and **movie genres/features**
- Context-aware recommendations: e.g., time of day, weekend
- Searchable **genre selection** for personalized recommendations
- Demo interface via **Streamlit**

---

## 📁 Dataset

- Dataset: **MovieLens 1M** ([link](https://grouplens.org/datasets/movielens/1m/))
- Data includes:
  - `movies.dat`: movie titles + genres
  - `ratings.dat`: user ratings
- Preprocessing:
  - Encode movie genres as **one-hot vectors**
  - Build **user-item interaction matrix**
  - Scale features as needed

---

## 🛠️ Installation

1. Clone repo:

```bash
git clone https://github.com/TungAnhDep/Hybrid-Recommendation-System.git
cd movie-recommender
```
2. Create environment
```bash
conda create -n movie_rec python=3.10
conda activate movie_rec
pip install -r requirements.txt
```
3. Demo
```bash
streamlit run app.py
```
-Enter User ID from 1 to 6039
-Choose time of the day/week, and genres
-Choose Top-N Recommendation

4. Train your own model

Download the MovieLens 1M dataset in the link above and run the script 
```bash
python train.py
```

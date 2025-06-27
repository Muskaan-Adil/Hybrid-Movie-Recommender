import pandas as pd

def load_data():
    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"]
    )
    ratings = pd.read_csv(
        "data/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )
    users = pd.read_csv(
        "data/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    )
    return movies, ratings, users

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_cf_model(ratings_df):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_cbf_model(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['Genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def hybrid_recommendations(user_id, cf_model, cbf_sim, movies_df, ratings_df, top_n=10):
    # get movies user has not rated
    seen = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].tolist()
    all_movies = movies_df['MovieID'].tolist()
    unseen = list(set(all_movies) - set(seen))
    
    # collaborative filtering predictions
    cf_scores = []
    for movie_id in unseen:
        cf_scores.append((movie_id, cf_model.predict(user_id, movie_id).est))
    cf_scores = sorted(cf_scores, key=lambda x: x[1], reverse=True)
    
    # take top 50 from CF to re-rank with CBF
    top_cf = cf_scores[:50]
    
    final_scores = []
    for movie_id, score in top_cf:
        idx = movies_df[movies_df['MovieID'] == movie_id].index[0]
        # simple: mean of CF + avg similarity to genres
        content_score = cbf_sim[idx].mean()
        hybrid_score = 0.7 * score + 0.3 * content_score
        final_scores.append((movie_id, hybrid_score))
    
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies_df[movies_df['MovieID'] == mid]['Title'].values[0] for mid, _ in final_scores[:top_n]]
    return top_movies

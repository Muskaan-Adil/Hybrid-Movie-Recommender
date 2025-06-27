import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="ISO-8859-1"
    )
    ratings = pd.read_csv(
        "data/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="ISO-8859-1"
    )
    users = pd.read_csv(
        "data/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="ISO-8859-1"
    )
    return movies, ratings, users

def train_cf_model(ratings_df):
    user_item_matrix = ratings_df.pivot_table(index="UserID", columns="MovieID", values="Rating").fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    return user_item_matrix, user_similarity

def train_cbf_model(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['Genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def hybrid_recommendations(user_id, user_item_matrix, user_similarity, cbf_sim, movies_df, ratings_df, top_n=10):
    # Collaborative filtering part
    user_idx = user_id - 1  # assuming UserIDs are 1-indexed
    similar_users = user_similarity[user_idx]
    weighted_ratings = similar_users @ user_item_matrix
    weighted_ratings /= similar_users.sum()
    
    user_seen = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].tolist()
    recommendations = []
    for movie_id in user_item_matrix.columns:
        if movie_id not in user_seen:
            cf_score = weighted_ratings[movie_id]
            try:
                idx = movies_df[movies_df['MovieID'] == movie_id].index[0]
                cb_score = cbf_sim[idx].mean()
                hybrid_score = 0.7 * cf_score + 0.3 * cb_score
            except:
                hybrid_score = cf_score
            recommendations.append((movie_id, hybrid_score))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [mid for mid, _ in recommendations[:top_n]]
    recommended_titles = movies_df[movies_df['MovieID'].isin(recommended_movie_ids)]['Title'].tolist()
    return recommended_titles

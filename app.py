import streamlit as st
import pickle
from recommender import load_data, hybrid_recommendations
from tmdb_api import get_poster

st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")

with st.spinner("Loading data and models..."):
    movies_df, ratings_df, users_df = load_data()
    with open("user_item_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)
    with open("user_similarity.pkl", "rb") as f:
        user_similarity = pickle.load(f)
    with open("cbf_sim.pkl", "rb") as f:
        cbf_sim = pickle.load(f)

user_ids = users_df['UserID'].unique().tolist()
selected_user = st.selectbox("Choose a User ID", user_ids)

if st.button("Get Recommendations"):
    recs = hybrid_recommendations(selected_user, user_item_matrix, user_similarity, cbf_sim, movies_df, ratings_df, top_n=5)
    st.subheader("Recommended Movies:")
    cols = st.columns(5)
    for idx, movie in enumerate(recs):
        with cols[idx]:
            st.markdown(f"**{movie}**")
            poster = get_poster(movie)
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.write("No poster found.")

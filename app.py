import streamlit as st
from recommender import load_data, train_cf_model, train_cbf_model, hybrid_recommendations
from tmdb_api import get_poster

st.title("🎬 Hybrid Movie Recommendation System")

with st.spinner("Loading data..."):
    movies_df, ratings_df, users_df = load_data()
    cf_model = train_cf_model(ratings_df)
    cbf_sim = train_cbf_model(movies_df)

user_ids = users_df['UserID'].unique().tolist()
selected_user = st.selectbox("Select User ID", user_ids)

if st.button("Recommend"):
    recs = hybrid_recommendations(selected_user, cf_model, cbf_sim, movies_df, ratings_df, top_n=5)
    for movie in recs:
        st.subheader(movie)
        poster = get_poster(movie)
        if poster:
            st.image(poster, width=200)
        else:
            st.write("Poster not found.")

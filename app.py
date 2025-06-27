import streamlit as st
from recommender import load_data, train_cf_model, train_cbf_model, hybrid_recommendations
from tmdb_api import get_poster

st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")

with st.spinner("Loading data and training models, please wait..."):
    movies_df, ratings_df, users_df = load_data()
    user_item_matrix, user_similarity = train_cf_model(ratings_df)
    cbf_sim = train_cbf_model(movies_df)

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

import pickle
from recommender import load_data, train_cf_model, train_cbf_model

print("Loading data and training models...")

movies_df, ratings_df, users_df = load_data()
user_item_matrix, user_similarity = train_cf_model(ratings_df)
cbf_sim = train_cbf_model(movies_df)

# Save them
with open("user_item_matrix.pkl", "wb") as f:
    pickle.dump(user_item_matrix, f)

with open("user_similarity.pkl", "wb") as f:
    pickle.dump(user_similarity, f)

with open("cbf_sim.pkl", "wb") as f:
    pickle.dump(cbf_sim, f)

print("✅ Models trained and saved.")

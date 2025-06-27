# Detailed Report – Hybrid Movie Recommendation System

## 1. Data Loading & Initial Exploration

The project began by loading three MovieLens `.dat` files (`movies.dat`, `ratings.dat`, and `users.dat`) into Pandas DataFrames. An initial inspection of the data structure and quality was performed:

* Displayed the shape and head of each DataFrame to confirm correct parsing and encoding.
* Checked for missing or inconsistent entries using `df.isnull().sum()` and basic `df.describe()` on numerical columns (like ratings).
* Verified that essential columns (UserID, MovieID, Rating, Title, and Genres) had no critical missing values.
* Cleaned up categorical fields by stripping whitespace and standardizing casing on genre and title columns.
* Confirmed that UserIDs and MovieIDs were consistent across the three data files.

This step ensured a reliable starting point for downstream modeling.

---

## 2. Feature Engineering

To enrich the hybrid recommendation system, several new features and transformations were applied:

* **User-Item Interaction Matrix**: Built a pivot table of users vs. movies with ratings, representing implicit user preference.
* **User Similarity Matrix**: Applied cosine similarity on the interaction matrix to identify similar user profiles for collaborative filtering.
* **TF-IDF Features**: Transformed the movie titles and genre strings into a TF-IDF weighted matrix to capture content-based similarity between movies.
* **Hybrid Score Calculation**: Combined user-user similarity and item-item similarity (from content) to create a hybrid recommendation score, giving more robust recommendations than a single method.
* **Poster Integration**: Added a feature that fetches movie posters dynamically via the TMDb API, enhancing user experience with richer visuals.

These engineered features allowed the recommender to handle both user preference patterns and item metadata for a hybrid recommendation output.

---

## 3. Feature Correlation & Similarity Analysis

To analyze the quality of the engineered similarities:

* Computed pairwise user similarity heatmaps for initial validation.
* Visualized genre distribution to ensure diverse coverage of movie types.
* Reviewed sparsity of the user-item matrix to check data coverage.
* Examined top recommendations to confirm logical consistency (for example, recommending sequels of the same series).

This step validated that both collaborative and content-based features aligned sensibly with user preferences.

---

## 4. Model Evaluation

The recommender system was evaluated by checking:

* **Coverage**: Number of movies recommended for each user
* **Novelty**: Whether recommended movies had previously been rated
* **Diversity**: Genre spread of recommended movies
* **Qualitative Checks**: Manually reviewed several recommendation outputs for plausibility

Evaluation was conducted on three configurations:

| Feature Set             | Description                                |
| ----------------------- | ------------------------------------------ |
| Collaborative Filtering | Based on user-user rating similarity       |
| Content-Based Filtering | Based on genre/title similarity via TF-IDF |
| Hybrid                  | Weighted combination of CF and CBF         |

The hybrid model showed the best balance of coverage, novelty, and diversity, improving user experience.

---

## 5. Deployment & Workflow

To make the project cloud-friendly:

* Pickle files were introduced (`user_item_matrix.pkl`, `user_similarity.pkl`, `cbf_sim.pkl`) to persist trained models.
* A fallback logic was added so the Streamlit app will re-train models on first launch if pickles are missing, supporting reproducibility and easy redeployment.
* Streamlit was chosen for its interactive UI capabilities, allowing real-time user selection and recommendations.
* TMDb API calls were rate-limited and wrapped with exception handling to gracefully manage missing posters.

---

## 6. Visual Summary

* The Streamlit dashboard displays a clean user dropdown, a “Get Recommendations” button, and a row of posters for the recommended movies.
* Posters and movie titles are arranged in a simple row of columns for readability.

---

## Final Thoughts

This project demonstrates the effectiveness of combining Collaborative Filtering with Content-Based Filtering for movie recommendations. By thoughtfully engineering hybrid features, storing models for reuse, and providing a Streamlit interface, I was able to:

* Increase the personalization of recommendations
* Reduce cold-start limitations common in pure CF
* Deliver an interactive, visually engaging app
* Simplify cloud deployment without large file uploads

---

### Future Enhancements

* Incorporate more metadata from TMDb, like director or cast, to enrich content features
* Test with implicit feedback data (e.g., clicks or watch time) instead of ratings alone
* Add user demographic-based filtering (age, gender, occupation)
* Explore more advanced hybrid models using matrix factorization or neural recommenders

The final hybrid pipeline offers a robust and flexible foundation for real-world movie recommendation systems.

# Hybrid Movie Recommendation System – Streamlit Deployment
---

## Project Overview

This project implements a hybrid movie recommendation system that combines Collaborative Filtering (CF) with Content-Based Filtering (CBF) techniques. The system leverages user ratings and movie metadata to provide more accurate and personalized recommendations. Additionally, the application uses the TMDb API to fetch movie posters, and is deployed on Streamlit for a user-friendly interface.

This project demonstrates how to build a real-world recommender by merging collaborative similarities with content features, then exposing the results in an interactive web app.

---

## Key Features

* **Data Loading**: Reads movie metadata, user data, and ratings from three .dat files.
* **Collaborative Filtering**: Calculates user-to-user similarities based on historical ratings.
* **Content-Based Filtering**: Measures similarities between movie metadata (genres, titles).
* **Hybrid Recommendation**: Blends both CF and CBF scores for stronger recommendations.
* **TMDb Poster Integration**: Fetches high-quality movie posters using the TMDb API.
* **Streamlit Deployment**: Presents a responsive UI with user selection, recommendations, and posters.
* **Model Persistence**: Stores trained similarity matrices as pickle files for faster reloads.
* **Self-Healing Training**: Automatically trains models on first deploy if pickles are missing.

---

## Dataset Information

**Source**: MovieLens dataset (commonly used in academic recommender systems).

**Files:**

* `movies.dat`: Contains movie titles and genre information.
* `ratings.dat`: Contains user ratings on movies.
* `users.dat`: Contains user demographic data.

**Key Columns:**

* `MovieID`: Unique ID for each movie.
* `UserID`: Unique ID for each user.
* `Rating`: User rating of the movie.
* `Title`: Movie title.
* `Genres`: Comma-separated genre list.
* `Gender`, `Age`, `Occupation`: Basic user demographics.

---

## Data Cleaning and Preprocessing Steps

### Loading the Dataset

* Loaded the three .dat files using pandas with appropriate separators.
* Checked for correct encoding and data consistency.

### Inspecting the Data

* Used `info()` to validate data types and spot nulls.
* Used `describe()` for basic numeric stats on the ratings.

### Handling Missing Values

* Verified that critical fields (MovieID, UserID, Rating) had no missing entries.
* Ensured movie metadata (genres, titles) was fully populated.

### Removing Duplicates

* Confirmed there were no duplicate user-movie-rating combinations in `ratings.dat`.

### Ensuring Data Consistency

* Standardized text fields with `.str.strip()` and consistent casing.
* Cross-checked user IDs and movie IDs for relational integrity.

### Date Formatting

* If timestamps were included, converted them to pandas `datetime` for future extension (not critical in current version).

---

## Model Training and Persistence

* Built a user-item interaction matrix from ratings.
* Computed cosine similarity between users for Collaborative Filtering.
* Created a TF-IDF matrix on movie titles/genres for Content-Based Filtering.
* Stored trained matrices (`user_item_matrix.pkl`, `user_similarity.pkl`, `cbf_sim.pkl`) for re-use.
* Added fallback in `app.py` to train models if no pickles exist, supporting easy cloud redeployment.

---

## Application UI

* Streamlit interface with dropdown to select UserID.
* On button click, shows top 5 recommended movies.
* Integrates movie posters from TMDb API for a richer visual experience.

---

## Data Visualizations

*(if you wish to expand in future)*

* You can add collaborative similarity heatmaps.
* Genre frequency histograms.
* User demographic distributions.

---

## Report

For more details about the hybrid recommendation logic, data transformation steps, and future enhancements, please refer to **HybridMovieRec\_Report.md**.

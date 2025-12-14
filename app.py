import streamlit as st
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# -------------------------------
# Load data
# -------------------------------
movies = pickle.load(open("movie_list.pkl", "rb"))

# -------------------------------
# Helper functions
# -------------------------------
def text_to_vector(text):
    return Counter(text.split())

@st.cache_data(show_spinner="Computing similarity matrix...")
def compute_similarity(movies_df):
    vectors = movies_df["tags"].apply(text_to_vector)

    # build vocabulary
    all_words = list(set(word for vec in vectors for word in vec))

    # convert to numeric matrix
    matrix = np.array([
        [vec.get(word, 0) for word in all_words]
        for vec in vectors
    ])

    # cosine similarity
    norm = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, matrix.T) / (norm[:, None] * norm[None, :])

    return similarity

# compute similarity (cached)
similarity = compute_similarity(movies)

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    for i in distances:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)



# import pickle
# import streamlit as st
# import requests
# import platform

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

# st.header('Movie Recommender System')
# import pickle

# movies = pickle.load(open("movie_list.pkl", "rb"))

# # from sklearn.metrics.pairwise import cosine_similarity
# # from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# from collections import Counter

# def text_to_vector(text):
#     return Counter(text.split())

# def cosine_similarity_matrix(matrix):
#     norm = np.linalg.norm(matrix, axis=1)
#     return np.dot(matrix, matrix.T) / (norm[:, None] * norm[None, :])

# # convert movie tags to vectors
# vectors = movies["tags"].apply(text_to_vector)

# # build vocabulary
# all_words = list(set(word for vec in vectors for word in vec))

# # convert to numeric matrix
# matrix = np.array([
#     [vec.get(word, 0) for word in all_words]
#     for vec in vectors
# ])

# # compute similarity
# similarity = cosine_similarity_matrix(matrix)

# # assuming movies is a DataFrame and has a 'tags' column
# cv = CountVectorizer(max_features=5000, stop_words='english')
# vectors = cv.fit_transform(movies['tags']).toarray()

# similarity = cosine_similarity(vectors)

# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     recommended_movie_names = []
#     recommended_movie_posters = []
#     for i in distances[1:6]:
#         # fetch the movie poster
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_posters.append(fetch_poster(movie_id))
#         recommended_movie_names.append(movies.iloc[i[0]].title)

#     return recommended_movie_names,recommended_movie_posters

# movie_list = movies['title'].values
# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )
# if st.button('Show Recommendation'):
#     recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(recommended_movie_names[0])
#         st.image(recommended_movie_posters[0])
#     with col2:
#         st.text(recommended_movie_names[1])
#         st.image(recommended_movie_posters[1])

#     with col3:
#         st.text(recommended_movie_names[2])
#         st.image(recommended_movie_posters[2])
#     with col4:
#         st.text(recommended_movie_names[3])
#         st.image(recommended_movie_posters[3])
#     with col5:
#         st.text(recommended_movie_names[4])
#         st.image(recommended_movie_posters[4])

# @st.cache_data
# def load_movies():
#     return pickle.load(open('movie_list.pkl', 'rb'))






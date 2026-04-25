import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(file_path):
    return pd.read_csv(file_path)

def calculate_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)


def recommend_movies(user_id, ratings_matrix, user_similarity):
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:]
    recommend_movies = {}
    for similar_user in similar_users.index:
        watched_movies = ratings_matrix.loc[similar_user][ratings_matrix.loc[similar_user] > 0]
        for movie, rating in watched_movies.items():
            if movie not in recommend_movies:
                recommend_movies[movie] = rating
            else:
                recommend_movies[movie] += rating

    return sorted(recommend_movies.items(), key=lambda x: x[1], reverse=True)

def main():

    ratings = load_dataset("movie_ratings.csv")
    ratings_matrix = ratings.pivot_table(index="user", columns="movie", values="rating").fillna(0)
    user_similarity = calculate_similarity(ratings_matrix)

    user_id = int(input("entre l'id de l'utilisateur : "))
    recommendations = recommend_movies(user_id, ratings_matrix, user_similarity)
    for movie, score in recommendations:
        print(f"{score:.2f} : {movie}")

if __name__ == "__main__":
    main()

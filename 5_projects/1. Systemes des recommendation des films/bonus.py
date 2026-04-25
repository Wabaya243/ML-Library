import numpy as np
import pandas as pd

data = {
    "book": ["Attention is all you need", "machine learning for beginners", "deep learning for beginners", "python for beginners", "datascience handbook",
     "articial intelligence guide", "python for data science", "deep learning fundamental"],
     "category" : ["AI", "ML", "DL", "programming", "DS", "AI", "DS", "DL"]
}

df = pd.DataFrame(data)
print(df)

user_rating = {
    "user": ["divin", "alice", "charlie", "david", "echo", "aline"],
    "Attention is all you need": [5, 4, 3, 2, 4, 3],
    "machine learning for beginners": [3, 4, 2, 3, 2, 4],
    "deep learning for beginners": [2, 3, 4, 2, 3, 4],
    "python for beginners": [4, 3, 2, 4, 3, 2],
    "datascience handbook": [3, 2, 4, 3, 2, 4],
    "articial intelligence guide": [2, 4, 3, 2, 4, 3],
    "python for data science": [4, 3, 2, 4, 3, 2],
    "deep learning fundamental": [3, 2, 4, 3, 2, 4]
}

df_user = pd.DataFrame(user_rating).set_index("user")
print(df_user)


# basé sur le contenu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = CountVectorizer()
category_vec = vectorizer.fit_transform(df["category"])

similarity_matrix = cosine_similarity(category_vec)
print(similarity_matrix)

def recommend_books(book):
    if book not in df["book"].values:
        print("ce livre n'est pas dans la base de donnée")
        return
    book_index = df[df["book"] == book].index[0]
    similarity_scores = list(enumerate(similarity_matrix[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_books = [df.iloc[i[0]]["book"] for i in similarity_scores[1:4]]
    return top_books

print(recommend_books("deep learning fundamental"))


# basé sur l'utilisateur
user_similarity_matrix = cosine_similarity(df_user.fillna(0))
print(user_similarity_matrix)
np.fill_diagonal(user_similarity_matrix, 0)
user_sim_df = pd.DataFrame(user_similarity_matrix, index=df_user.index, columns=df_user.index)

def recommend_books_user(user):
    if user not in df_user.index:
        print("ce utilisateur n'est pas dans la base de donnée")
        return
    similar_user = user_sim_df[user].sort_values(ascending=False).index[0]
    recommended_books = df_user.loc[similar_user][df_user.loc[similar_user] == 5].index.tolist()
    return recommended_books

print(recommend_books_user("divin"))

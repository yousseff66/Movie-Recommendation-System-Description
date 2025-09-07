import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# 1. تحميل البيانات
# ======================
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print("Ratings:")
print(ratings.head())
print("Movies:")
print(movies.head())

# نتاكد الأعمدة
# ratings فيه userId, movieId, rating
# movies فيه movieId, title, genres

# ======================
# 2. بناء user-item matrix
# ======================
user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_item_matrix = user_item_matrix.fillna(0)

# ======================
# 3. حساب التشابه بين المستخدمين
# ======================
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=user_item_matrix.index, 
                                  columns=user_item_matrix.index)

# ======================
# 4. دالة التوصية
# ======================
def recommend_movies(user_id, k=5):
    # المستخدمين الأكثر تشابهاً
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]  
    
    # الأفلام اللي شافها المستخدم
    user_movies = set(ratings[ratings.userId == user_id].movieId)
    
    # الأفلام من المستخدمين المشابهين
    recommendations = {}
    for other_user, sim in similar_users.items():
        other_movies = ratings[ratings.userId == other_user]
        for _, row in other_movies.iterrows():
            if row.movieId not in user_movies:  # فيلم جديد
                if row.movieId not in recommendations:
                    recommendations[row.movieId] = row.rating * sim
                else:
                    recommendations[row.movieId] += row.rating * sim
    
    # ترتيب الأفلام
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
    # رجع العناوين بدل IDs
    return movies[movies.movieId.isin([m for m, _ in recommended_movies])]["title"].tolist()

# تجربة
print("Recommendations for user 1:", recommend_movies(1, k=5))

# ======================
# 5. Precision@K
# ======================
def precision_at_k(user_id, k=5):
    recommended = recommend_movies(user_id, k)
    
    # الأفلام اللي قيمها المستخدم >= 3
    relevant = ratings[(ratings.userId == user_id) & (ratings.rating >= 3)]
    relevant_titles = movies[movies.movieId.isin(relevant.movieId)]["title"].tolist()
    
    if len(relevant_titles) == 0:
        return None
    
    hits = len([m for m in recommended if m in relevant_titles])
    return hits / k

# حساب المتوسط على أول 50 مستخدم
precisions = []
for user in ratings.userId.unique()[:50]:
    p = precision_at_k(user, k=5)
    if p is not None:
        precisions.append(p)

print("Average Precision@5:", np.mean(precisions))

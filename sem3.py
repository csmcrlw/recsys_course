"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)

        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"]).toarray()

        max_movie_id = int(self.movies_df["movieId"].max())
        n_genres = genre_matrix.shape[1]

        self.embeddings = np.zeros((max_movie_id + 1, n_genres))

        for row_idx, movie_id in enumerate(self.movies_df["movieId"]):
            self.embeddings[int(movie_id)] = genre_matrix[row_idx]

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        if item_id >= self.embeddings.shape[0]:
            return 0.0

        target_vec = self.embeddings[item_id]

        if np.linalg.norm(target_vec) == 0:
            return 0.0

        user_ratings = self.ui_matrix[user_id]
        rated_item_ids = np.where(user_ratings > 0)[0]

        if len(rated_item_ids) == 0:
            return 0.0

        rated_vectors = self.embeddings[rated_item_ids]
        rated_ratings = user_ratings[rated_item_ids]

        target_norm = np.linalg.norm(target_vec)
        rated_norms = np.linalg.norm(rated_vectors, axis=1)

        valid_mask = rated_norms > 0

        if not np.any(valid_mask):
            return 0.0

        rated_vectors = rated_vectors[valid_mask]
        rated_ratings = rated_ratings[valid_mask]
        rated_norms = rated_norms[valid_mask]

        similarities = rated_vectors @ target_vec / (rated_norms * target_norm)

        top_indices = np.argsort(similarities)[::-1][:k]

        top_similarities = similarities[top_indices]
        top_ratings = rated_ratings[top_indices]

        sim_sum = top_similarities.sum()

        if sim_sum == 0:
            return 0.0

        predicted_rating = np.dot(top_similarities, top_ratings) / sim_sum
        predicted_rating = np.clip(predicted_rating, 0.0, 5.0)

        return float(predicted_rating)

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        user_ratings = self.ui_matrix[user_id]
        rated_item_ids = np.where(user_ratings > 0)[0]

        if len(rated_item_ids) == 0:
            return []

        rated_vectors = self.embeddings[rated_item_ids]
        rated_ratings = user_ratings[rated_item_ids]

        valid_mask = np.linalg.norm(rated_vectors, axis=1) > 0

        if not np.any(valid_mask):
            return []

        rated_vectors = rated_vectors[valid_mask]
        rated_ratings = rated_ratings[valid_mask]

        user_profile = np.average(
            rated_vectors,
            axis=0,
            weights=rated_ratings,
        )

        profile_norm = np.linalg.norm(user_profile)

        if profile_norm == 0:
            return []

        movie_norms = np.linalg.norm(self.embeddings, axis=1)

        similarities = np.divide(
            self.embeddings @ user_profile,
            movie_norms * profile_norm,
            out=np.zeros(self.embeddings.shape[0], dtype=float),
            where=movie_norms > 0,
        )

        user_seen_mask = user_ratings > 0

        max_len = min(len(user_seen_mask), len(similarities))
        similarities[:max_len][user_seen_mask[:max_len]] = -1

        sorted_items = np.argsort(similarities)[::-1]

        recommendations = [
            int(item_id)
            for item_id in sorted_items
            if similarities[item_id] >= 0
        ]

        return recommendations[:n_recommendations]


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")


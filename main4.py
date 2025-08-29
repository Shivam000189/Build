import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, data):
        self.data = data

        # Fill missing values
        self.data["genre"] = self.data["genre"].fillna("")
        self.data["language"] = self.data["language"].fillna("")
        self.data["year"] = self.data["year"].fillna("")

        # TF-IDF Vectorizer (fit once on genre)
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.data["genre"])

        # Index movie names for quick lookup
        self.indices = pd.Series(self.data.index, index=self.data["movies_name"].str.lower()).drop_duplicates()

    def recommend(self, movie_name, top_n=5):
        movie_name = movie_name.lower()

        if movie_name not in self.indices:
            return f"❌ Movie '{movie_name}' not found in dataset."

        idx = self.indices[movie_name]

        # Compute similarity ONLY for this movie
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Get language & year of the input movie
        movie_language = self.data.loc[idx, "language"]
        movie_year = self.data.loc[idx, "year"]

        # Get scores sorted
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)

        # Filter: remove itself + keep only same language + same year
        sim_scores = [
            s for s in sim_scores 
            if s[0] != idx 
            and self.data.loc[s[0], "language"] == movie_language
            and self.data.loc[s[0], "year"] == movie_year
        ][:top_n]

        movie_indices = [i[0] for i in sim_scores]

        return self.data.iloc[movie_indices][["movies_name", "genre", "language", "year", "rating_10"]]

# =====================
# Main Program
# =====================
if __name__ == "__main__":
    movies = pd.read_csv("movie.csv")

    system = MovieRecommender(movies)

    user_input = input("Enter a movie name: ")

    recs = system.recommend(user_input, top_n=5)
    print("\nRecommended Movies:")
    print(recs.to_string(index=False))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, books_df):
        self.books_df = books_df
        # Ensure data types are correct
        self.books_df['Book-Title'] = self.books_df['Book-Title'].astype(str)
        self.books_df['Book-Author'] = self.books_df['Book-Author'].astype(str)
        self.books_df['Publisher'] = self.books_df['Publisher'].astype(str)
        
        self.cosine_sim = None
        self.indices = None

    def preprocess(self):
        # Create a 'features' column by combining relevant text fields
        # usage of fillna to handle potentially missing values although we cast to str in init
        self.books_df['features'] = (
            self.books_df['Book-Title'] + " " + 
            self.books_df['Book-Author'] + " " + 
            self.books_df['Publisher']
        )
        # Clean up the features (basic example)
        self.books_df['features'] = self.books_df['features'].apply(lambda x: x.lower())

    def train(self):
        # Initialize TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Construct the TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(self.books_df['features'])
        
        # Compute the cosine similarity matrix
        # linear_kernel is equivalent to cosine_similarity for normalized vectors (TF-IDF is normalized)
        # and is faster
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Construct a reverse map of indices and book titles
        # We use the first occurrence of a book title to map to its index
        # Note: In a real scenario with duplicate titles, we might want to use ISBN, 
        # but users search by title.
        self.indices = pd.Series(self.books_df.index, index=self.books_df['Book-Title']).drop_duplicates()

    def recommend(self, title, top_n=5):
        # Check if book exists
        if title not in self.indices:
            return []

        # Get the index of the book that matches the title
        idx = self.indices[title]
        
        # Handle case where multiple books have the same title (if drop_duplicates didn't catch it logic wise)
        if isinstance(idx, pd.Series):
             idx = idx.iloc[0]

        # Get the pairwsie similarity scores of all books with that book
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar books
        # We start from 1 because 0 is the book itself
        sim_scores = sim_scores[1:top_n+1]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Return the top_n most similar books
        return self.books_df.iloc[book_indices]

    def get_books_by_genre(self, genre, top_n=10):
        """
        Simulate genre filtering by searching keywords in Book-Title.
        """
        keywords = {
            'Fantasy': ['magic', 'wizard', 'dragon', 'fantasy', 'ring', 'harry potter', 'lord of the rings', 'hobbit', 'witch'],
            'Mystery': ['mystery', 'detective', 'murder', 'crime', 'sherlock', 'poirot', 'investigation', 'thriller'],
            'Romance': ['love', 'romance', 'kiss', 'wedding', 'bride', 'heart'],
            'Sci-Fi': ['space', 'planet', 'alien', 'galaxy', 'star wars', 'scifi', 'sci-fi', 'robot', 'future'],
            'Horror': ['horror', 'ghost', 'vampire', 'zombie', 'scary', 'haunted', 'stephen king']
        }
        
        if genre not in keywords:
            return pd.DataFrame() # Return empty if genre not defined
            
        search_terms = keywords[genre]
        # Regex pattern to match any of the keywords case-insensitively
        pattern = '|'.join(search_terms)
        
        # Filter books where title contains the pattern
        filtered_books = self.books_df[self.books_df['Book-Title'].str.contains(pattern, case=False, na=False)]
        
        # Return a random sample if we have more than top_n, else return all we found
        if len(filtered_books) > top_n:
            return filtered_books.sample(top_n)
        return filtered_books

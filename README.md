# Content-Based Book Recommendation System

## 📌 Project Objective
This project is a Machine Learning application designed to recommend books to users based on content similarity. Unlike simple popularity-based systems, this engine analyzes book titles, authors, and publishers using Natural Language Processing (NLP) techniques to find the most relevant matches.

## 📂 Dataset Used
The system uses the **Book-Crossing Dataset** (subset):
- `BX-Books.csv`: Contains book information (ISBN, Title, Author, Year, Publisher).
- `BX-Users.csv`: Anonymous user data (ID, Location, Age) - *Used for future hybrid filtering*.
- `BX-Book-Ratings-Subset.csv`: User ratings - *Used for future hybrid filtering*.

## 🧠 Algorithm: TF-IDF & Cosine Similarity
The core recommendation logic is based on **Content-Based Filtering**:

1.  **Feature Engineering**: We combine `Book-Title`, `Book-Author`, and `Publisher` into a single text feature.
2.  **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency):
    - Converts text data into numerical vectors.
    - Lowers the weight of common words (stopwords) and highlights unique keywords.
3.  **Cosine Similarity**:
    - Calculates the angle between vectors to determine similarity.
    - A score of 1.0 means identical content, while 0.0 means no similarity.

## 🚀 How to Run the Project

### Prerequisites
- Python 3.12+
- Dependencies: `pandas`, `streamlit`, `scikit-learn`

### Installation
1.  Navigate to the project directory:
    ```bash
    cd Book-Recommender-master-master
    ```
2.  Install dependencies:
    ```bash
    pip install .
    ```

### Execution
Run the Streamlit application:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🔮 Future Scope
- **Hybrid Filtering**: Combine Collaborative Filtering (User Ratings) with Content-Based Filtering.
- **Deep Learning**: Use Neural Networks for feature extraction.
- **Enhanced UI**: Add more interactive elements and book reviews.

---
*Academic Project Submission*

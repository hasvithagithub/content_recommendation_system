import streamlit as st
import pandas as pd
import template as t
from recommender import ContentBasedRecommender

st.set_page_config(page_title="Book Recommender", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING & CACHING ---
@st.cache_resource
def load_data():
    try:
        # Load books
        books = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
        
        # Load ratings for Top 50 functionality
        ratings = pd.read_csv('data/BX-Book-Ratings-Subset.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
        
        return books, ratings
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'data/BX-Books.csv' and 'data/BX-Book-Ratings-Subset.csv' exist.")
        return None, None

@st.cache_resource
def load_recommender(data):
    # We use a subset of data for performance in this demo if the dataset is huge
    # For full project, use full data or better hardware/optimization
    # Taking top 5000 books for speed and memory efficiency in Cloud environment
    if len(data) > 5000:
        data_subset = data.head(5000).copy()
    else:
        data_subset = data.copy()
        
    model = ContentBasedRecommender(data_subset)
    model.preprocess()
    model.train()
    return model

@st.cache_data
def get_top_50_books(books_df, ratings_df):
    # Aggregating ratings
    rating_counts = ratings_df.groupby('ISBN').count()['Book-Rating'].reset_index()
    rating_counts.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    
    avg_rating = ratings_df.groupby('ISBN').mean(numeric_only=True)['Book-Rating'].reset_index()
    avg_rating.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    
    # Merging
    popular_df = rating_counts.merge(avg_rating, on='ISBN')
    
    # Filter for statistically significant ratings (e.g., > 50 votes)
    # Adjust threshold if data is sparse, let's try 50 first, if empty we lower it
    popular_df = popular_df[popular_df['num_ratings'] >= 50]
    
    # Merge with Books to get titles and images
    popular_df = popular_df.merge(books_df, on='ISBN')
    
    # Sort by rating
    popular_df = popular_df.sort_values('avg_rating', ascending=False).head(50)
    
    return popular_df

# --- PAGE FUNCTIONS ---

def page_recommend_books(df_books):
    st.title("Search & Recommend")
    st.markdown("Enter a book title below to get personalized recommendations.")

    # Initialize Recommender
    with st.spinner('Training AI model...'):
        recommender = load_recommender(df_books)

    # User Input
    # sophisticated autocomplete
    all_titles = recommender.books_df['Book-Title'].unique()
    selected_book = st.selectbox(
        "Type or select a book you like:",
        options=[""] + list(all_titles),
        index=0,
        help="Start typing to search for a book title."
    )

    if selected_book:
        st.success(f"Selected: **{selected_book}**")
        
        # Display selected book info
        try:
            original_book_info = recommender.books_df[recommender.books_df['Book-Title'] == selected_book].iloc[0]
            
            st.write("---")
            st.markdown(
                f"""
                <div class="book-card" style="flex-direction: row; align-items: flex-start; text-align: left; padding: 25px; margin-bottom: 30px; gap: 20px;">
                    <img src="{original_book_info['Image-URL-M']}" style="width: 140px; height: 210px; object-fit: cover; box-shadow: 0 5px 15px rgba(0,0,0,0.5);">
                    <div style="flex: 1;">
                        <h3 style="margin: 0 0 10px 0; font-size: 1.8rem; line-height: 1.2;">{original_book_info['Book-Title']}</h3>
                        <p style="color: #cbd5e1; font-size: 1.1rem; margin-bottom: 5px;"><strong>Author:</strong> {original_book_info['Book-Author']}</p>
                        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 5px;"><strong>Publisher:</strong> {original_book_info['Publisher']}</p>
                        <p style="color: #94a3b8; font-size: 0.95rem;"><strong>Year:</strong> {original_book_info['Year-Of-Publication']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("### Recommended Books:")
            
            recommendations = recommender.recommend(selected_book, top_n=5)
            
            if not isinstance(recommendations, list) and not recommendations.empty:
                t.recommendations(recommendations)
            else:
                st.warning("No recommendations found or book not in index.")
                
        except IndexError:
            st.error("Error retrieving book details.")

    # --- GENRE BROWSING ---
    st.write("---")
    st.subheader("Explore by Genre")
    
    genres = ['Fantasy', 'Mystery', 'Romance', 'Sci-Fi', 'Horror']
    selected_genre = st.selectbox("Select a Genre", genres)
    
    if selected_genre:
        st.markdown(f"### Top Picks in {selected_genre}")
        with st.spinner(f"Finding best {selected_genre} books..."):
            genre_predictions = recommender.get_books_by_genre(selected_genre, top_n=10)
            
        if not genre_predictions.empty:
            t.recommendations(genre_predictions)
        else:
            st.info(f"No books found matching criteria for {selected_genre} in this subset of data.")

def page_top_50(df_books, df_ratings):
    st.title("Top 50 Books")
    st.markdown("Here are the top 50 highly rated books by our community.")
    
    with st.spinner('Crunching numbers...'):
        top_books = get_top_50_books(df_books, df_ratings)
        
    if top_books.empty:
        st.warning("Not enough rating data to calculate Top 50 (threshold: 50 votes).")
        return

    # Display in a grid with dividers
    cols_per_row = 4
    books = top_books.reset_index().to_dict('records')
    
    # Chunk the books into rows
    rows = [books[i:i + cols_per_row] for i in range(0, len(books), cols_per_row)]
    
    for i, row_items in enumerate(rows):
        cols = st.columns(cols_per_row)
        
        for idx, book in enumerate(row_items):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="book-card">
                        <img src="{book['Image-URL-M']}" style="width: 120px; height: 170px; object-fit: cover;">
                        <div class="book-title" title="{book['Book-Title']}">{book['Book-Title']}</div>
                        <div class="book-author">{book['Book-Author']}</div>
                        <div class="book-stats">
                            ★ {book['avg_rating']:.1f} | {book['num_ratings']} Votes
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Add a divider after each row, except the last one
        if i < len(rows) - 1:
            st.markdown("---")

def page_about():
    st.title("About Project")
    st.markdown(
        """
        ### Book Recommendation System
        
        This project is a **Content-Based Recommendation System** developed using Streamlit and Python.
        
        #### Features:
        - **Personalized Recommendations**: Based on book titles using TF-IDF and Cosine Similarity.
        - **Genre Exploration**: Browse books by popular genres.
        - **Top 50 Books**: A curated list of the highest-rated books in our dataset.
        
        #### Technologies Used:
        - **Streamlit**: For the web interface.
        - **Pandas**: For data manipulation.
        - **Scikit-learn**: For machine learning algorithms.
        
        #### Credits:
        Dataset provided by the Book-Crossing community.
        """
    )


# --- MAIN APP LOGIC ---

def main():
    # Sidebar Navigation
    st.sidebar.title("Book Recommender")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Top 50 Books", "Recommend Books", "About"],
        index=1 # Default to Recommender as per implicit user flow, or change if needed
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Navigate through the app using the menu above.")

    # Dark Mode Toggle
    dark_mode = st.sidebar.checkbox("Night Vision Mode", value=False)
    
    # Load Data
    with st.spinner('Loading library...'):
        df_books, df_ratings = load_data()
    
    if df_books is None:
        return

    # Routing
    if page == "Top 50 Books":
        if df_ratings is not None:
            page_top_50(df_books, df_ratings)
        else:
            st.error("Ratings data is missing, cannot display Top 50.")
            
    elif page == "Recommend Books":
        page_recommend_books(df_books)
        
    elif page == "About":
        page_about()

    # --- CUSTOM CSS ---
    # Theme Variables
    if dark_mode:
        # Premium Dark Mode (Glassmorphism + Deep Ocean)
        root_vars = """
        :root {
            --primary-color: #8b5cf6;
            --secondary-color: #06b6d4;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --sidebar-bg: #111827;
            --sidebar-text: #ffffff;
            --input-bg: #334155;
            --input-border: #475569;
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        /* Dark Mode Overrides */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--input-border);
        }
        .stRadio label, .stCheckbox label, .stMarkdown label, .stSelectbox label {
            color: var(--sidebar-text) !important;
        }
        /* MAIN CONTAINER (Fix for white background in Dark Mode) */
        .stApp, html, body {
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
        }
        """
    else:
        # Premium Light Mode (Clean Slate + Indigo)
        root_vars = """
        :root {
            --primary-color: #6366f1; /* Indigo-500 */
            --secondary-color: #0ea5e9; /* Sky-500 */
            --bg-color: #ffffff; 
            --card-bg: #ffffff;
            --text-color: #0f172a; /* Slate-900 */
            --sidebar-bg: #f8fafc; /* Slate-50 */
            --sidebar-text: #334155; /* Slate-700 */
            --input-bg: #f8fafc;
            --input-border: #e2e8f0;
            --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        """
    
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        {root_vars}

        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
            /* background-color handles in variable block for specific override */
        }}

        /* HEADERS */
        h1, h2, h3 {{
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }}

        /* CUSTOM BUTTONS */
        .stButton>button {{
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            border-radius: 25px;
            border: none;
            padding: 10px 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(var(--primary-color), 0.3);
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(var(--primary-color), 0.5);
        }}

        /* INPUT FIELDS */
        .stTextInput>div>div>input, .stSelectbox>div>div>div {{
            background-color: var(--input-bg) !important; 
            color: var(--text-color) !important;
            border: 1px solid var(--input-border);
            border-radius: 12px;
        }}
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
        }}

        /* CUSTOM SIDEBAR RADIO BUTTONS */
        /* Target the radio button container in sidebar */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] {{
            background-color: transparent !important;
            border-left: 4px solid transparent !important; 
            padding-left: 10px !important;
            margin-bottom: 5px;
            transition: all 0.2s ease;
            color: var(--sidebar-text) !important; /* Force text color based on mode */
        }}

        /* Selected State Styling */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child input:checked + div {{
            font-weight: 600 !important;
            color: var(--sidebar-text) !important; /* Force selected text color based on mode */
        }}

        /* Force all text inside radio buttons to be the sidebar text color */
        section[data-testid="stSidebar"] .stRadio p, 
        section[data-testid="stSidebar"] .stRadio div,
        section[data-testid="stSidebar"] .stRadio span,
        section[data-testid="stSidebar"] .stRadio label {{
            color: var(--sidebar-text) !important;
        }}
        
        /* The container logic for the selected item - using :has if supported or generic checked sibling approach */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {{
             border-left: 5px solid var(--primary-color) !important;
             background-color: rgba(139, 92, 246, 0.1) !important; /* Soft purple bg */
        }}
        
        /* Fallback for browsers not supporting :has (though most modern ones do) - 
           Streamlit puts 'aria-checked' on the element wrapping the text often, but let's try broadly */
        div[role="radiogroup"] label[aria-checked="true"] {{
            border-left: 5px solid var(--primary-color) !important;
            background-color: rgba(139, 92, 246, 0.1) !important;
        }}

        /* Force Radio Circle Color to Primary */
        section[data-testid="stSidebar"] .stRadio input[type="radio"] {{
            accent-color: var(--primary-color) !important;
            filter: hue-rotate(240deg); /* Attempt to shift red to purple/blue if accent-color fails */
        }}
        /* More specific targetting for the circle if it's a pseudo element in some themes */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {{
             /* This is usually the circle container */
        }}

        /* CARDS (Used in both Top 50 and Recommendations) */
        .book-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 15px;
            border: 1px solid var(--input-border);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            box-shadow: var(--shadow);
            margin-bottom: 25px; /* Gap between rows */
            color: var(--text-color);
        }}
        .book-card:hover {{
            transform: translateY(-5px);
            border-color: var(--primary-color);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }}
        .book-card img {{
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 12px;
            transition: transform 0.3s ease;
        }}
        .book-card:hover img {{
            transform: scale(1.05);
        }}
        .book-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 5px;
            color: var(--text-color);
            /* clamp text to 2 lines */
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            height: 3.4em;
        }}
        .book-author {{
            font-size: 0.9rem;
            color: var(--text-color);
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        .book-stats {{
            margin-top: auto;
            font-size: 0.85rem;
            color: var(--secondary-color);
            font-weight: 600;
            background: rgba(6, 182, 212, 0.1);
            padding: 4px 12px;
            border-radius: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Footer
    st.write("---")
    st.markdown(
        """
        <div style='text-align: center; color: #64748b; padding: 20px;'>
            <small>Designed with ❤️ | Academic Project | ML Powered</small>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
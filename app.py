import streamlit as st
import pandas as pd
import template as t
from recommender import ContentBasedRecommender

st.set_page_config(page_title="Book Recommender", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING & CACHING ---
@st.cache_resource
def load_data():
    try:
        # Load only necessary columns to save memory if needed, but for now loading all
        # Adjust paths if necessary
        books = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
        # We don't necessarily need Ratings/Users for strictly content-based, 
        # unless we want to display extra info. For now, just books.
        return books
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'data/BX-Books.csv' exists.")
        return None

@st.cache_resource
def load_recommender(data):
    # We use a subset of data for performance in this demo if the dataset is huge
    # For full project, use full data or better hardware/optimization
    # Taking top 10000 books for speed in 'mini project' scope
    if len(data) > 20000:
        data_subset = data.head(20000).copy()
    else:
        data_subset = data.copy()
        
    model = ContentBasedRecommender(data_subset)
    model.preprocess()
    model.train()
    return model

# --- MAIN APP LOGIC ---

def main():
    # Sidebar
    st.sidebar.title("Book Recommender")
    st.sidebar.markdown("### Discover your next favorite read!")
    st.sidebar.info("This system uses **TF-IDF** and **Cosine Similarity** to find books with similar content (titles, authors, publishers).")

    # Main Area
    st.title("Search & Recommend")
    st.markdown("Enter a book title below to get personalized recommendations.")

    # Load Data
    with st.spinner('Loading library...'):
        df_books = load_data()
    
    if df_books is None:
        return

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
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(original_book_info['Image-URL-M'], width=130)
            with col2:
                st.subheader(original_book_info['Book-Title'])
                st.markdown(f"**Author:** {original_book_info['Book-Author']}")
                st.markdown(f"**Publisher:** {original_book_info['Publisher']}")
                st.caption(f"Year: {original_book_info['Year-Of-Publication']}")

            st.write("### Recommended Books:")
            
            recommendations = recommender.recommend(selected_book, top_n=5)
            
            if not isinstance(recommendations, list) and not recommendations.empty:
                t.recommendations(recommendations)
            else:
                st.warning("No recommendations found or book not in index.")
                
        except IndexError:
            st.error("Error retrieving book details.")
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
    # --- CUSTOM CSS ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Modern Button Styling */
        .stButton>button {
            border-radius: 20px;
            font-weight: 600;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Dark Mode Toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    dark_mode = st.sidebar.checkbox("Night Vision Mode", value=False)
    
    if dark_mode:
        st.markdown(
            """
            <style>
            /* Main App Background - Deep Blue-Grey */
            .stApp {
                background-color: #0f1116;
            }
            
            /* Sidebar Background - Slightly Lighter */
            section[data-testid="stSidebar"] {
                background-color: #161b22;
                border-right: 1px solid #30363d;
            }
            
            /* Typography Colors */
            h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, p, label {
                color: #c9d1d9 !important;
            }
            .stCaption {
                color: #8b949e !important;
            }
            
            /* Input Fields */
            .stTextInput > div > div > input, .stSelectbox > div > div > div {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            
            /* Cards/Images Background (if applicable) */
            img {
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            
            /* Success/Info Messages */
            .stSuccess, .stInfo {
                background-color: #1f2937 !important;
                color: #e5e7eb !important;
                border: 1px solid #374151;
            }
            
            /* Button Override for Dark Mode */
            .stButton>button {
                background-color: #238636;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.write("---")
    st.markdown(
        """
        <div style='text-align: center; color: grey;'>
            <small>Academic Project | Content-Based Filtering | ML</small>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
import streamlit as st
from random import random

# set episode session state
# Helper functions for UI
# Helper functions for UI
def tile_item(column, item):
    with column:
        st.markdown(
            f"""
            <div class="book-card">
                <img src="{item['Image-URL-M']}" style="width: 100%; height: 200px; object-fit: cover;">
                <div class="book-title" title="{item['Book-Title']}">{item['Book-Title']}</div>
                <div class="book-author">{item['Book-Author']}</div>
                <div class="book-stats">
                    {item['Year-Of-Publication']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def recommendations(df):
    # check the number of items
    nbr_items = df.shape[0]

    if nbr_items != 0:    
        # Create a grid layout
        # We'll display 5 items per row
        cols_per_row = 5
        rows = [df.iloc[i:i+cols_per_row] for i in range(0, nbr_items, cols_per_row)]

        for row_df in rows:
            columns = st.columns(cols_per_row)
            items = row_df.to_dict(orient='records')
            
            # If the last row has fewer items, we zip only up to that length
            for col, item in zip(columns, items):
                tile_item(col, item)
    else:
        st.info("No recommendations available for this selection.") 


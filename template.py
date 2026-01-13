import streamlit as st
from random import random

# set episode session state
# Helper functions for UI
def tile_item(column, item):
    with column:
        # Use a container for better spacing and alignment
        with st.container():
            # Display image with a fixed width to ensure uniformity if possible, 
            # though st.image 'use_column_width' is usually good responsiveness.
            try:
                st.image(item['Image-URL-M'], use_container_width=True)
            except:
                st.text("No Image")
                
            st.markdown(f"**{item['Book-Title']}**")
            st.caption(f"{item['Book-Author']}")
            st.caption(f"{item['Year-Of-Publication']}")

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


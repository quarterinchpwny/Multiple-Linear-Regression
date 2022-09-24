import MultiP
import LinearP
import Corr
import streamlit as st
PAGES = {
    "Multiple Linear Regression": MultiP,
    "Linear Regression": LinearP,
    "Correlation Analysis": Corr
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
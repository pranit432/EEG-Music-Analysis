"""Streamlit interface for the EEG-and-music project.

Run from the project folder:  python -m streamlit run src/app.py
"""
import streamlit as st

import views

st.set_page_config(page_title="EEG & Music", page_icon=":material/neurology:", layout="wide",
                   menu_items={"About": "Band-power analysis of EEG recorded while people listen to music at different tempos."})

page_explore = st.Page(views.explore_page, title="Explore a recording", icon=":material/search:", url_path="explore", default=True)
views.PAGE_EXPLORE = page_explore

st.title("Effect of Music on EEG Brain Waves")
st.sidebar.caption("Band-power analysis of 18 subjects listening to music at different tempos.\n\n"
                   "Data: Kaggle, Music Tempo Emotional Impact EEG Dataset.")
st.navigation([
    page_explore,
    st.Page(views.group_page, title="Baseline vs music", icon=":material/compare_arrows:", url_path="group"),
    st.Page(views.listen_page, title="Listen", icon=":material/headphones:", url_path="listen"),
    st.Page(views.about_page, title="About", icon=":material/info:", url_path="about"),
]).run()

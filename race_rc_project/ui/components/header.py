"""Header component for the app"""

import streamlit as st

def render_header():
    """Render main app header."""
    st.markdown("""
    <div class="main-header">[QUIZ] Reading Comprehension Quiz Generator</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Intelligently Generate and Verify Reading Comprehension Quizzes**
    
    Powered by:
    - Model A: 10-classifier ensemble (LR, SVM, NB, RF, XGB, Voting, Stacking, K-Means, Label Propagation, GMM)
    - Model B: Distractor & Hint Generator (Word2Vec + Similarity Ranking)
    """)
    st.divider()

def render_screen_title(screen_number, screen_name):
    """Render a screen title."""
    st.markdown(f'<div class="screen-title">Screen {screen_number}: {screen_name}</div>', 
                unsafe_allow_html=True)

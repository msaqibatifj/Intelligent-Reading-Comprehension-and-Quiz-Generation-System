"""Styling utilities for the app"""

import streamlit as st

def apply_styles():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        :root{
            --primary: #D72638; /* red */
            --secondary: #1A1A1B; /* black */
            --text: #F5F5F5; /* off-white */
            --panel: rgba(255,255,255,0.03);
        }
        body, .stApp {
            background-color: var(--secondary) !important;
            color: var(--text) !important;
        }
        .main-header {
            font-size: 2.5em;
            color: var(--primary);
            margin-bottom: 10px;
        }
        .screen-title {
            font-size: 1.8em;
            color: var(--primary);
            margin: 20px 0;
            border-bottom: 3px solid var(--primary);
            padding-bottom: 10px;
        }
        .quiz-option {
            padding: 15px;
            margin: 10px 0;
            border: 2px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            background-color: transparent;
            color: var(--text);
        }
        .quiz-option:hover {
            border-color: var(--primary);
            background-color: rgba(215,38,56,0.08);
        }
        .quiz-option.selected {
            border-color: var(--primary);
            background-color: var(--primary);
            color: var(--text);
        }
        .hint-box {
            background-color: var(--panel);
            border-left: 4px solid var(--primary);
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            color: var(--text);
        }
        .metric-box {
            background-color: var(--panel);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
            color: var(--text);
        }
        .prediction-correct {
            color: #2ca02c;
            font-weight: bold;
        }
        .prediction-incorrect {
            color: #d62728;
            font-weight: bold;
        }
        /* ensure Streamlit metrics and widgets inherit text color */
        .stMetric, .stMarkdown, .stText, .stStreamlitWidget {
            color: var(--text) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def section_divider():
    """Render a visual divider."""
    st.divider()

def metric_card(label, value, delta=None):
    """Render a metric card."""
    col = st.container()
    with col:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(label, value, delta=delta)
        st.markdown('</div>', unsafe_allow_html=True)

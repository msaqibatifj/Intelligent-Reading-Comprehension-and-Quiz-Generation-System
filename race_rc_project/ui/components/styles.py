"""Styling utilities for the app"""

import streamlit as st

def apply_styles():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5em;
            color: #1f77b4;
            margin-bottom: 10px;
        }
        .screen-title {
            font-size: 1.8em;
            color: #2ca02c;
            margin: 20px 0;
            border-bottom: 3px solid #2ca02c;
            padding-bottom: 10px;
        }
        .quiz-option {
            padding: 15px;
            margin: 10px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .quiz-option:hover {
            border-color: #1f77b4;
            background-color: #f0f7ff;
        }
        .quiz-option.selected {
            border-color: #1f77b4;
            background-color: #1f77b4;
            color: white;
        }
        .hint-box {
            background-color: #fff8dc;
            border-left: 4px solid #ff7f0e;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .metric-box {
            background-color: #f0f7ff;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .prediction-correct {
            color: #2ca02c;
            font-weight: bold;
        }
        .prediction-incorrect {
            color: #d62728;
            font-weight: bold;
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

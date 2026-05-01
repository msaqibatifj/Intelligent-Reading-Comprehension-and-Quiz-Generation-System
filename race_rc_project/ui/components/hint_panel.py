"""Hint Panel component (Screen 3)"""

import streamlit as st
from .header import render_screen_title

def render_hint_panel():
    """Render Screen 3: Hint Panel."""
    render_screen_title(3, "Hint Panel")
    
    # Display correct answer
    correct_answer = st.session_state.options[st.session_state.correct_answer]
    st.markdown(f"**Correct Answer:** {correct_answer}")
    
    # Display hints
    if st.session_state.hints:
        render_hints_list()
    else:
        st.info("No hints available. Generate hints from Quiz View.")
    
    # Navigation buttons
    render_hint_panel_buttons()

def render_hints_list():
    """Render list of hints."""
    st.markdown("### Generated Hints:")
    
    for idx, hint in enumerate(st.session_state.hints, 1):
        if isinstance(hint, dict):
            hint_text = hint.get('text', str(hint))
            hint_score = hint.get('score', 0)
            st.markdown(f"""
            <div class="hint-box">
                <strong>Hint {idx}</strong> (Quality: {hint_score:.2f})
                <br>{hint_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hint-box">
                <strong>Hint {idx}</strong>
                <br>{hint}
            </div>
            """, unsafe_allow_html=True)

def render_hint_panel_buttons():
    """Render action buttons for hint panel."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Back to Quiz", use_container_width=True, key="btn_hint_back_quiz"):
            st.session_state.screen = 'quiz_view'
            st.rerun()
    
    with col2:
        if st.button("View Analytics", use_container_width=True, key="btn_hint_analytics"):
            st.session_state.screen = 'analytics'
            st.rerun()
    
    with col3:
        if st.button("Back to Input", use_container_width=True, key="btn_hint_back_input"):
            st.session_state.screen = 'article_input'
            st.rerun()

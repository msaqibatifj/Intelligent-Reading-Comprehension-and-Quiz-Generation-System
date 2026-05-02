"""Quiz View component (Screen 2)"""

import streamlit as st
from .header import render_screen_title
from .utils import get_option_letter, set_screen


def _fallback_hints(article: str, correct_answer: str, num_hints: int = 3):
    """Generate simple fallback hints from the article text."""
    sentences = [s.strip() for s in article.split('.') if s.strip()]
    hints = []

    for sentence in sentences:
        if correct_answer and correct_answer.lower() in sentence.lower():
            continue
        hints.append({
            'text': sentence,
            'score': max(0.5, 0.85 - (0.05 * len(hints))),
            'source': 'fallback'
        })
        if len(hints) >= num_hints:
            break

    if not hints:
        hints = [{
            'text': 'Review the passage for key details related to the question.',
            'score': 0.5,
            'source': 'fallback'
        }]

    return hints

def render_quiz_view():
    """Render Screen 2: Quiz View."""
    render_screen_title(2, "Quiz View")
    
    # Display article
    with st.expander("Show Article", expanded=False):
        st.text_area(
            "Article:",
            value=st.session_state.article,
            disabled=True,
            height=120,
            key="quiz_view_article_display"
        )
    
    # Display question
    st.markdown(f"### Question: {st.session_state.question}")
    
    # Display options
    render_quiz_options()
    
    # Navigation buttons
    render_quiz_view_buttons()

def render_quiz_options():
    """Render the 4 quiz options."""
    st.markdown("### Select Your Answer:")
    
    option_labels = ["A", "B", "C", "D"]
    
    for idx, (label, option_text) in enumerate(zip(option_labels, st.session_state.options)):
        col1, col2 = st.columns([1, 20])
        
        with col1:
            if st.button(label, key=f"opt_btn_{idx}", use_container_width=True):
                st.session_state.user_answer = idx
        
        with col2:
            # Highlight if selected
            bg_style = "background-color: #1f77b4; color: white;" if st.session_state.user_answer == idx else ""
            st.markdown(
                f'<div style="padding: 15px; border: 2px solid #ddd; border-radius: 8px; {bg_style}">{option_text}</div>', 
                unsafe_allow_html=True
            )

def render_quiz_view_buttons():
    """Render action buttons for quiz view."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Back to Input", use_container_width=True, key="btn_back_to_input"):
            set_screen('article_input')
            st.rerun()
    
    with col2:
        if st.button("Show Hints", use_container_width=True, key="btn_show_hints"):
            inference = st.session_state.get('inference')
            correct_answer_text = st.session_state.options[st.session_state.correct_answer]

            hints = []
            if inference:
                hints_result = inference.generate_hints(
                    correct_answer=correct_answer_text,
                    article=st.session_state.article,
                    num_hints=3
                )
                if isinstance(hints_result, dict):
                    hints = hints_result.get('hints', [])

            if not hints:
                hints = _fallback_hints(
                    st.session_state.article,
                    correct_answer_text,
                    num_hints=3
                )

            st.session_state.hints = hints
            set_screen('hint_panel')
            st.rerun()
    
    with col3:
        if st.button("Check Answer", use_container_width=True, key="btn_check_answer"):
            if st.session_state.user_answer is None:
                st.error("Please select an answer first!")
            else:
                set_screen('analytics')
                st.rerun()
    
    with col4:
        if st.button("Submit & Next", use_container_width=True, key="btn_submit_next"):
            if st.session_state.user_answer is None:
                st.error("Please select an answer!")
            else:
                st.success("Quiz submitted! Check analytics for results.")

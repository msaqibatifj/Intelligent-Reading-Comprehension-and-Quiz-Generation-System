"""Quiz View component (Screen 2)"""

import streamlit as st
from .header import render_screen_title
from .utils import get_option_letter, set_screen, get_active_question_data, set_question_index


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
    """Render Screen 2: Quiz View — one question at a time."""
    render_screen_title(2, "Quiz View")

    active_question, current_index, total_questions = get_active_question_data()

    # --- Question navigation (one-by-one) ---
    if total_questions > 1:
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅ Previous", key="quiz_prev_question"):
                # Reset answer reveal when switching questions
                st.session_state.answer_revealed = False
                st.session_state.user_answer = None
                set_question_index(max(0, current_index - 1))
                st.rerun()
        with nav_col2:
            st.markdown(f"**Question {current_index + 1} of {total_questions}**")
        with nav_col3:
            if st.button("Next ➡", key="quiz_next_question"):
                st.session_state.answer_revealed = False
                st.session_state.user_answer = None
                set_question_index(min(total_questions - 1, current_index + 1))
                st.rerun()

        # Progress bar
        st.progress((current_index + 1) / total_questions)
    
    # Display article
    with st.expander("📖 Show Article", expanded=False):
        st.text_area(
            "Article:",
            value=st.session_state.article,
            disabled=True,
            height=120,
            key="quiz_view_article_display"
        )
    
    # Display question
    st.markdown(f"### ❓ Question: {active_question.get('question', st.session_state.question)}")

    # Keep the active answer/options aligned with the selected question bundle when available.
    if 'answer' in active_question and 'options' in active_question:
        st.session_state.question = active_question['question']
        st.session_state.options = active_question['options']
        st.session_state.correct_answer = active_question['correct_answer']
    
    # Display options
    render_quiz_options()

    # --- Answer reveal section ---
    render_answer_section(active_question)
    
    # Navigation buttons
    render_quiz_view_buttons(active_question)

def render_quiz_options():
    """Render the 4 quiz options."""
    st.markdown("### Select Your Answer:")
    
    option_labels = ["A", "B", "C", "D"]
    answer_revealed = st.session_state.get('answer_revealed', False)
    correct_idx = st.session_state.correct_answer
    
    for idx, (label, option_text) in enumerate(zip(option_labels, st.session_state.options)):
        col1, col2 = st.columns([1, 20])
        
        with col1:
            if st.button(label, key=f"opt_btn_{idx}", use_container_width=True):
                st.session_state.user_answer = idx
                st.rerun()
        
        with col2:
            # Determine styling based on state
            if answer_revealed:
                if idx == correct_idx:
                    bg_style = "background-color: #28a745; color: white; border-color: #28a745;"
                    indicator = " ✅"
                elif st.session_state.user_answer == idx:
                    bg_style = "background-color: #dc3545; color: white; border-color: #dc3545;"
                    indicator = " ❌"
                else:
                    bg_style = "opacity: 0.6;"
                    indicator = ""
            else:
                if st.session_state.user_answer == idx:
                    bg_style = "background-color: #1f77b4; color: white;"
                    indicator = ""
                else:
                    bg_style = ""
                    indicator = ""
            
            st.markdown(
                f'<div style="padding: 15px; border: 2px solid #ddd; border-radius: 8px; {bg_style}">{option_text}{indicator}</div>', 
                unsafe_allow_html=True
            )


def render_answer_section(active_question):
    """Render the Show Answer button and answer feedback."""
    answer_revealed = st.session_state.get('answer_revealed', False)

    if not answer_revealed:
        if st.button("🔍 Show Answer", key="btn_show_answer", use_container_width=True):
            st.session_state.answer_revealed = True
            st.rerun()
    else:
        # Show detailed answer feedback
        correct_idx = st.session_state.correct_answer
        correct_text = st.session_state.options[correct_idx]
        user_answer = st.session_state.user_answer

        st.markdown("---")
        if user_answer is not None and user_answer == correct_idx:
            st.success(f"🎉 Correct! The answer is **{get_option_letter(correct_idx)}: {correct_text}**")
        elif user_answer is not None:
            user_text = st.session_state.options[user_answer]
            st.error(f"❌ Incorrect. You chose **{get_option_letter(user_answer)}: {user_text}**")
            st.info(f"✅ The correct answer is **{get_option_letter(correct_idx)}: {correct_text}**")
        else:
            st.info(f"✅ The correct answer is **{get_option_letter(correct_idx)}: {correct_text}**")

        # Show source sentence if available
        source = active_question.get('source_sentence', '')
        if source:
            with st.expander("📝 Source from passage"):
                st.markdown(f"*\"{source}\"*")


def render_quiz_view_buttons(active_question):
    """Render action buttons for quiz view."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅ Back to Input", use_container_width=True, key="btn_back_to_input"):
            st.session_state.answer_revealed = False
            set_screen('article_input')
            st.rerun()
    
    with col2:
        if st.button("💡 Show Hints", use_container_width=True, key="btn_show_hints"):
            correct_answer_text = active_question.get('answer', st.session_state.options[st.session_state.correct_answer])

            hints = []
            inference = st.session_state.get('inference')
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
            st.session_state.hints_revealed = 0
            set_screen('hint_panel')
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True, key="btn_check_answer"):
            set_screen('analytics')
            st.rerun()


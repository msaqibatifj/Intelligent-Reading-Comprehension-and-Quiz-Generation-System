"""Utility functions for the app"""

import streamlit as st

def validate_quiz_input(article, question, options, correct_answer):
    """Validate that all quiz input fields are filled."""
    errors = []
    
    if not article or not article.strip():
        errors.append("Article is required")
    
    if not question or not question.strip():
        errors.append("Question is required")
    
    if not all(opt and opt.strip() for opt in options):
        errors.append("All 4 options are required")
    
    if correct_answer is None:
        errors.append("Correct answer must be selected")
    
    return len(errors) == 0, errors

def get_option_letter(index):
    """Convert index to option letter (0->A, 1->B, etc)."""
    return chr(65 + index)

def get_option_index(letter):
    """Convert option letter to index (A->0, B->1, etc)."""
    return ord(letter.upper()) - ord('A')

def format_confidence(confidence):
    """Format confidence score as percentage."""
    if confidence is None:
        return "N/A"
    return f"{confidence:.1%}"

def check_user_answer(user_answer_idx, correct_answer_idx):
    """Check if user answer is correct."""
    if user_answer_idx is None:
        return None
    return user_answer_idx == correct_answer_idx


def set_screen(screen_name: str) -> None:
    """Update the active screen."""
    st.session_state.screen = screen_name


def get_active_question_data():
    """Return the currently active generated question data, or a single-question fallback."""
    question_bundles = st.session_state.get('question_bundles', [])
    generated_questions = st.session_state.get('generated_questions', [])
    current_index = st.session_state.get('current_question_index', 0)

    if question_bundles:
        current_index = max(0, min(current_index, len(question_bundles) - 1))
        return question_bundles[current_index], current_index, len(question_bundles)

    if generated_questions:
        current_index = max(0, min(current_index, len(generated_questions) - 1))
        return generated_questions[current_index], current_index, len(generated_questions)

    return {
        'question': st.session_state.get('question', ''),
        'source_sentence': st.session_state.get('article', ''),
        'template_type': 'single',
    }, 0, 1


def set_question_index(index: int) -> None:
    """Update the active question index within the generated set."""
    st.session_state.current_question_index = index

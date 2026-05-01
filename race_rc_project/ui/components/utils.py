"""Utility functions for the app"""

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

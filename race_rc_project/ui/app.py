"""
Main Streamlit app entry point.
Uses component-based architecture for clean separation of concerns.

Screens:
1. Article Input - Load article, question, options
2. Quiz View - Display quiz, collect user answer
3. Hint Panel - Generate hints for correct answer
4. Analytics Dashboard - Model predictions & confidence
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
from components import (
    render_header,
    render_sidebar,
    render_article_input,
    render_quiz_view,
    render_hint_panel,
    render_analytics_dashboard,
    apply_styles,
)

from src.inference import UnifiedInference

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Reading Comprehension Quiz Generator",
    page_icon="[QUIZ]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Apply Styling
# ============================================================================

apply_styles()

# ============================================================================
# Load Inference Models
# ============================================================================

@st.cache_resource
def load_models():
    """Load unified inference models."""
    try:
        base_dir = Path(__file__).parent.parent
        # Define model paths
        model_a_paths = {
            'lr': base_dir / 'models/model_a/traditional/lr_model.pkl',
            'svm': base_dir / 'models/model_a/traditional/svm_model.pkl',
            'nb': base_dir / 'models/model_a/traditional/nb_model.pkl',
            'rf': base_dir / 'models/model_a/traditional/rf_model.pkl',
            'xgb': base_dir / 'models/model_a/traditional/xgb_model.pkl',
            'ensemble_voting': base_dir / 'models/model_a/traditional/ensemble_voting_model.pkl',
            'ensemble_stacking': base_dir / 'models/model_a/traditional/ensemble_stacking_model.pkl',
            'kmeans': base_dir / 'models/model_a/traditional/kmeans_model.pkl',
            'label_propagation': base_dir / 'models/model_a/traditional/label_propagation_model.pkl',
            'gmm': base_dir / 'models/model_a/traditional/gmm_model.pkl',
            'feature_engineer': base_dir / 'models/model_a/traditional/feature_engineer.pkl',
        }
        
        model_b_paths = {
            'distractor_hint_generator': base_dir / 'models/model_b/traditional/distractor_hint_generator.pkl',
            'word2vec': base_dir / 'models/model_b/traditional/word2vec_model.pkl',
        }
        
        inference = UnifiedInference(model_a_paths, model_b_paths)
        if getattr(inference, 'load_errors', None):
            errors = inference.load_errors
            if errors.get('model_a') or errors.get('model_b'):
                st.warning("Some models failed to load. See details below.")
                st.json(errors)
        return inference
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'screen' not in st.session_state:
        st.session_state.screen = 'article_input'
    
    if 'article' not in st.session_state:
        st.session_state.article = ""
    
    if 'question' not in st.session_state:
        st.session_state.question = ""
    
    if 'options' not in st.session_state:
        st.session_state.options = ["", "", "", ""]
    
    if 'correct_answer' not in st.session_state:
        st.session_state.correct_answer = 0
    
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = None
    
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = None
    
    if 'hints' not in st.session_state:
        st.session_state.hints = []

    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []

    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0

    if 'answer_revealed' not in st.session_state:
        st.session_state.answer_revealed = False

    if 'hints_revealed' not in st.session_state:
        st.session_state.hints_revealed = 0
    
    if 'inference' not in st.session_state or st.session_state.inference is None:
        st.session_state.inference = load_models()

initialize_session_state()

# ============================================================================
# Main App
# ============================================================================

def main():
    """Main app router - orchestrates component rendering."""
    
    # Render header
    render_header()
    
    # Render sidebar and handle navigation
    render_sidebar()
    
    # Route to appropriate screen based on session state
    screen = st.session_state.screen
    
    if screen == 'article_input':
        render_article_input()
    elif screen == 'quiz_view':
        render_quiz_view()
    elif screen == 'hint_panel':
        render_hint_panel()
    elif screen == 'analytics':
        render_analytics_dashboard()
    else:
        st.error(f"Unknown screen: {screen}")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **[QUIZ] - Intelligent Reading Comprehension & Quiz Generation**
    
    **Components-based Architecture**
    - ui/components/header.py - Header and titles
    - ui/components/sidebar.py - Navigation and state display
    - ui/components/article_input.py - Screen 1
    - ui/components/quiz_view.py - Screen 2
    - ui/components/hint_panel.py - Screen 3
    - ui/components/analytics_dashboard.py - Screen 4
    - ui/components/styles.py - Custom CSS
    - ui/components/utils.py - Shared utilities
    """)

if __name__ == "__main__":
    main()

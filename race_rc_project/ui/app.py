"""
Streamlit UI for Intelligent Reading Comprehension and Quiz Generation System.
4 Screens: Article Input, Quiz View, Hint Panel, Analytics Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference import UnifiedInference, ModelAInference, ModelBInference
from evaluate import ModelAEvaluator, ModelBEvaluator

# Page configuration
st.set_page_config(
    page_title="Reading Comprehension & Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .correct {
        color: #28a745;
        font-weight: bold;
    }
    .incorrect {
        color: #dc3545;
        font-weight: bold;
    }
    .hint-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_article' not in st.session_state:
    st.session_state.current_article = ""
if 'current_mcq' not in st.session_state:
    st.session_state.current_mcq = None
if 'user_answer' not in st.session_state:
    st.session_state.user_answer = None
if 'hints_revealed' not in st.session_state:
    st.session_state.hints_revealed = []
if 'session_results' not in st.session_state:
    st.session_state.session_results = []
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None


# Initialize models (mock for now)
@st.cache_resource
def load_inference_engine():
    """Load trained models."""
    model_a_paths = {
        'lr': 'models/model_a/traditional/lr_model.pkl',
        'svm': 'models/model_a/traditional/svm_model.pkl',
        'rf': 'models/model_a/traditional/rf_model.pkl',
        'ensemble': 'models/model_a/traditional/ensemble_model.pkl',
    }
    
    model_b_paths = {
        'distractor_ranker': 'models/model_b/traditional/distractor_ranker.pkl',
        'hint_scorer': 'models/model_b/traditional/hint_scorer.pkl',
    }
    
    return UnifiedInference(model_a_paths, model_b_paths)


# ============================================================================
# SCREEN 1: Article Input
# ============================================================================
def screen_article_input():
    st.header("📄 Article Input")
    st.write("Paste a reading passage or upload a text file to generate quiz questions.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        article_text = st.text_area(
            "Paste reading passage:",
            height=300,
            placeholder="Enter or paste your reading passage here...",
            key="article_textarea"
        )
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("📚 Load Sample (RACE)", use_container_width=True):
            sample_text = """
            Alice in Wonderland is a novel by Lewis Carroll, first published in 1865. 
            The story follows a young girl named Alice who falls down a rabbit hole 
            and enters a fantastical world filled with peculiar characters and situations. 
            The novel is famous for its imaginative storytelling, wordplay, and the way 
            it explores themes of identity, logic, and the nature of reality. Carroll's 
            work has become a cornerstone of children's literature and continues to 
            inspire adaptations, analyses, and discussions nearly 160 years after its publication.
            """
            st.session_state.current_article = sample_text
            st.rerun()
        
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file:
            article_text = uploaded_file.read().decode('utf-8')
    
    if st.button("✨ Generate Quiz Question", use_container_width=True, type="primary"):
        if not article_text:
            st.error("❌ Please enter a passage first!")
        else:
            st.session_state.current_article = article_text
            
            # Simulate model inference
            with st.spinner("🔄 Generating quiz..."):
                time.sleep(2)  # Mock inference time
                
                # Mock MCQ generation
                st.session_state.current_mcq = {
                    'question': "What is the main theme of Alice in Wonderland?",
                    'correct_answer': "Themes of identity, logic, and the nature of reality",
                    'distractors': [
                        "A love story between Alice and the Cheshire Cat",
                        "A guide to Victorian etiquette",
                        "Instructions for cooking exotic tea"
                    ],
                    'hints': [
                        "Think about what the novel explores about the world.",
                        "The author was interested in logic and mathematics.",
                        "Alice constantly questions what is real and who she is."
                    ]
                }
                st.session_state.hints_revealed = []
                st.success("✓ Quiz question generated!")
            
            st.rerun()


# ============================================================================
# SCREEN 2: Quiz View
# ============================================================================
def screen_quiz_view():
    st.header("❓ Quiz Question")
    
    if st.session_state.current_mcq is None:
        st.warning("⚠ No question generated yet. Go to 'Article Input' first.")
        return
    
    mcq = st.session_state.current_mcq
    
    # Display article context
    with st.expander("📖 View Passage", expanded=False):
        st.text_area("Passage:", value=st.session_state.current_article, height=200, disabled=True)
    
    # Display question
    st.subheader(mcq['question'])
    
    # Display options
    st.write("### Choose your answer:")
    options = mcq['distractors'] + [mcq['correct_answer']]
    np.random.seed(42)  # For reproducibility in demo
    shuffled_idx = np.random.permutation(len(options))
    options_shuffled = [options[i] for i in shuffled_idx]
    
    answer_idx = st.radio(
        "Options:",
        range(len(options_shuffled)),
        format_func=lambda i: options_shuffled[i],
        label_visibility="collapsed"
    )
    
    selected_answer = options_shuffled[answer_idx]
    
    # Check answer button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✔ Check Answer", use_container_width=True, type="primary"):
            is_correct = selected_answer == mcq['correct_answer']
            
            if is_correct:
                st.success("✅ **Correct!**", icon="✅")
            else:
                st.error("❌ **Incorrect!**", icon="❌")
                st.info(f"**Correct answer:** {mcq['correct_answer']}")
            
            # Log result
            st.session_state.session_results.append({
                'question': mcq['question'],
                'user_answer': selected_answer,
                'correct_answer': mcq['correct_answer'],
                'is_correct': is_correct,
                'timestamp': pd.Timestamp.now()
            })
    
    with col2:
        if st.button("🔄 Generate New Question", use_container_width=True):
            st.session_state.current_mcq = None
            st.rerun()


# ============================================================================
# SCREEN 3: Hint Panel
# ============================================================================
def screen_hint_panel():
    st.header("💡 Hint Panel")
    
    if st.session_state.current_mcq is None:
        st.warning("⚠ No question available. Generate a quiz first.")
        return
    
    hints = st.session_state.current_mcq['hints']
    
    st.write("Hints will help you without revealing the answer directly.")
    
    for i, hint in enumerate(hints):
        col1, col2 = st.columns([0.8, 0.2])
        
        with col1:
            if i < len(st.session_state.hints_revealed):
                st.markdown(f"""
                <div class="hint-box">
                    <strong>Hint {i+1}:</strong> {hint}
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(f"Reveal Hint {i+1}"):
                    st.session_state.hints_revealed.append(i)
                    st.rerun()
    
    # Reveal answer (only after all hints)
    if len(st.session_state.hints_revealed) == len(hints):
        st.divider()
        if st.button("🎯 Reveal Answer", use_container_width=True, type="secondary"):
            st.markdown(f"**Answer:** {st.session_state.current_mcq['correct_answer']}")


# ============================================================================
# SCREEN 4: Analytics Dashboard
# ============================================================================
def screen_analytics():
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.session_results:
        st.info("📈 No quiz results yet. Answer some questions to see analytics.")
        return
    
    results_df = pd.DataFrame(st.session_state.session_results)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_answered = len(results_df)
    correct_count = results_df['is_correct'].sum()
    accuracy = (correct_count / total_answered * 100) if total_answered > 0 else 0
    
    with col1:
        st.metric("Total Questions", total_answered)
    with col2:
        st.metric("Correct", correct_count)
    with col3:
        st.metric("Accuracy", f"{accuracy:.1f}%")
    with col4:
        st.metric("Avg Time", "5.2s")  # Mock
    
    st.divider()
    
    # Model A Metrics
    st.subheader("📋 Model A - Q&A Verification")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision", f"{0.89:.2f}")  # Mock
    with col2:
        st.metric("Recall", f"{0.85:.2f}")  # Mock
    with col3:
        st.metric("F1 Score", f"{0.87:.2f}")  # Mock
    
    # Model B Metrics
    st.subheader("📋 Model B - Distractor & Hint Generation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Distractor Accuracy", f"{0.91:.2f}")  # Mock
    with col2:
        st.metric("Hint Precision", f"{0.88:.2f}")  # Mock
    with col3:
        st.metric("Avg Inference Time", "2.1s")  # Mock
    
    st.divider()
    
    # Results table
    st.subheader("📊 Question-by-Question Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Export
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="quiz_results.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================================
# Main Navigation
# ============================================================================
def main():
    st.title("📚 Intelligent Reading Comprehension & Quiz Generator")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Article Input", "Quiz View", "Hint Panel", "Analytics Dashboard"]
    )
    
    # Display selected screen
    if page == "Article Input":
        screen_article_input()
    elif page == "Quiz View":
        screen_quiz_view()
    elif page == "Hint Panel":
        screen_hint_panel()
    elif page == "Analytics Dashboard":
        screen_analytics()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 1.0.0  \n**Status:** In Development")


if __name__ == "__main__":
    main()

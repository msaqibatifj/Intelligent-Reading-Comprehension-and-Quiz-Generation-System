"""Article Input component (Screen 1)"""

import streamlit as st
from .header import render_screen_title
from .utils import validate_quiz_input, set_screen, set_question_index
from .sample_data import (
    load_ai_generated_mode,
    load_model_generated_questions_mode,
    load_race_dataset_mode,
    load_user_provided_mode,
    get_random_article,
)


def render_article_input():
    """Render Screen 1: Article Input with 3 modes."""
    render_screen_title(1, "Article Input")
    
    # Initialize mode if not set
    if 'quiz_mode' not in st.session_state:
        st.session_state.quiz_mode = 'user_provided'
    
    # Handle deferred article loading (must happen BEFORE widgets render)
    if 'pending_load' in st.session_state and st.session_state.pending_load:
        if st.session_state.pending_load == 'random':
            load_random()
        elif st.session_state.pending_load == 'example':
            load_example()
        st.session_state.pending_load = None
        st.rerun()
    
    # Mode selection tabs
    st.markdown("### Choose Quiz Mode")
    mode = st.radio(
        "Select how you want to create the quiz:",
        options=['ai_generated', 'model_generated', 'user_provided'],
        format_func=lambda x: {
            'ai_generated': '[SIMPLE] AI generates question using templates',
            'model_generated': '[3-STEP PIPELINE] Model A generates using NLP + ranking',
            'user_provided': '[CUSTOM] Full control • You write everything'
        }[x],
        horizontal=False,
        key="mode_selector"
    )
    
    # Update mode
    if mode != st.session_state.quiz_mode:
        st.session_state.quiz_mode = mode
        st.rerun()
    
    st.divider()
    
    # Render mode-specific UI
    if st.session_state.quiz_mode == 'ai_generated':
        render_ai_generated_mode()
    elif st.session_state.quiz_mode == 'model_generated':
        render_model_generated_mode()
    else:  # user_provided
        render_user_provided_mode()

def render_options_input():
    """Render the 4 option input fields."""
    st.markdown("### Options")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        opt_a = st.text_input(
            "Option A:",
            value=st.session_state.options[0],
            key="opt_a"
        )
        st.session_state.options[0] = opt_a
    
    with col_b:
        opt_b = st.text_input(
            "Option B:",
            value=st.session_state.options[1],
            key="opt_b"
        )
        st.session_state.options[1] = opt_b
    
    with col_c:
        opt_c = st.text_input(
            "Option C:",
            value=st.session_state.options[2],
            key="opt_c"
        )
        st.session_state.options[2] = opt_c
    
    with col_d:
        opt_d = st.text_input(
            "Option D:",
            value=st.session_state.options[3],
            key="opt_d"
        )
        st.session_state.options[3] = opt_d

def render_correct_answer_selection():
    """Render correct answer selection."""
    st.markdown("### Correct Answer")
    correct = st.radio(
        "Select the correct answer:",
        options=["A", "B", "C", "D"],
        index=st.session_state.correct_answer,
        horizontal=True,
        key="radio_correct_answer"
    )
    st.session_state.correct_answer = ord(correct) - ord('A')

def render_article_input_buttons():
    """Render action buttons for article input screen."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Next to Quiz", key="btn_next_quiz", use_container_width=True):
            is_valid, errors = validate_quiz_input(
                st.session_state.article,
                st.session_state.question,
                st.session_state.options,
                st.session_state.correct_answer
            )
            
            if not is_valid:
                st.error("\n".join(errors))
            else:
                set_screen('quiz_view')
                st.rerun()
    
    with col2:
        if st.button("Load Example", use_container_width=True):
            load_example()
            st.rerun()
    
    with col3:
        if st.button("Load Random Trial", use_container_width=True):
            load_random()
            st.rerun()

def load_example():
    """Load example article (Great Wall of China).
    
    User will need to write their own question to test Model A verification.
    """
    st.session_state.article = (
        "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, "
        "and wood, built along the historical northern borders of China to protect against various nomadic groups. "
        "It is one of the most iconic structures in the world. Construction began as early as the 7th century BC "
        "and continued for over 2,000 years."
    )
    # Update widget-specific keys too
    st.session_state.ai_gen_article = st.session_state.article
    st.session_state.model_gen_article = st.session_state.article
    st.session_state.user_article = st.session_state.article
    st.session_state.question = ""
    st.session_state.options = ["", "", "", ""]
    st.session_state.correct_answer = 0
    st.session_state.generated_questions = []
    st.session_state.current_question_index = 0


def load_random():
    """Load random article from sample dataset for trial.
    
    User will need to write their own question to test Model A verification.
    """
    article = get_random_article()
    st.session_state.article = article
    # Update widget-specific keys too
    st.session_state.ai_gen_article = article
    st.session_state.model_gen_article = article
    st.session_state.user_article = article
    st.session_state.question = ""
    st.session_state.options = ["", "", "", ""]
    st.session_state.correct_answer = 0
    st.session_state.generated_questions = []
    st.session_state.current_question_index = 0
    st.session_state.question_bundles = []


# =============================================================================
# Mode 1: AI-Generated Questions
# =============================================================================

def render_ai_generated_mode():
    """MODE 1: AI generates question, user writes answer."""
    st.info("""
    ### Mode 1: AI-Generated Questions
    1. **System generates question** using AI templates
    2. **You write answer and distractors** (3 wrong options)
    3. **Model A verifies** if Q&A is valid
    4. **Model B ranks distractors** and generates hints
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Article")
        article = st.text_area(
            "Paste or load article:",
            value=st.session_state.article,
            height=150,
            placeholder="Enter article text...",
            key="ai_gen_article"
        )
        st.session_state.article = article
    
    with col2:
        st.markdown("### Quick Load")
        if st.button("Load Random", use_container_width=True, key="ai_gen_load"):
            st.session_state.pending_load = 'random'
            st.rerun()
        
        if st.button("Load Example", use_container_width=True, key="ai_gen_example"):
            st.session_state.pending_load = 'example'
            st.rerun()
    
    # Generate question from AI
    if st.session_state.article:
        if st.button("Generate Question", type="primary", use_container_width=True, key="ai_gen_btn"):
            from .sample_data import generate_question_ai
            question, hint = generate_question_ai(st.session_state.article)
            st.session_state.question = question
            st.info(f"💡 Hint: {hint}")
            st.rerun()
    
    # Question display
    st.markdown("### Generated Question")
    st.text_input(
        "Question:",
        value=st.session_state.question,
        disabled=True,
        key="ai_gen_question_display"
    )
    
    # User writes answer and distractors
    st.markdown("### You Write Answer & Options")
    
    answer = st.text_input(
        "Correct Answer:",
        placeholder="Write the answer to the question",
        key="ai_gen_answer"
    )
    st.session_state.options[0] = answer
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        opt_a = st.text_input("Wrong Option 1:", key="ai_gen_opt_a")
        st.session_state.options[1] = opt_a
    
    with col_b:
        opt_b = st.text_input("Wrong Option 2:", key="ai_gen_opt_b")
        st.session_state.options[2] = opt_b
    
    with col_c:
        opt_c = st.text_input("Wrong Option 3:", key="ai_gen_opt_c")
        st.session_state.options[3] = opt_c
    
    st.session_state.correct_answer = 0  # User's answer is always correct
    
    # Proceed button
    if st.button("Next to Quiz", type="primary", use_container_width=True, key="ai_gen_next"):
        if not all([st.session_state.article, st.session_state.question, st.session_state.options[0]]):
            st.error("Please fill in article, question, and answer!")
        else:
            set_screen('quiz_view')
            st.rerun()


# =============================================================================
# Mode 2: RACE Dataset
# =============================================================================

def render_model_generated_mode():
    """MODE 2: Generate Questions using Model A 3-Step Pipeline."""
    st.info("""
    ### Mode 2: Model A Generates Questions (3-Step Pipeline)
    **STEP 1:** Extract important sentences (TF-IDF keyword overlap)
    **STEP 2:** Generate Wh-question candidates (templates)
    **STEP 3:** Rank with Model A (SVM/RF ensemble) → pick best
    
    1. **Model A generates question** automatically
    2. **You write answer and distractors** (or verify generated)
    3. **Model A evaluates** Q&A quality
    4. **Model B improves** distractors and generates hints
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Article")
        article = st.text_area(
            "Paste your article:",
            value=st.session_state.article,
            height=150,
            placeholder="Enter article text...",
            key="model_gen_article"
        )
        st.session_state.article = article
    
    with col2:
        st.markdown("### Quick Load")
        if st.button("Random Article", use_container_width=True, key="model_gen_random"):
            st.session_state.pending_load = 'random'
            st.rerun()
        
        if st.button("Example", use_container_width=True, key="model_gen_example"):
            st.session_state.pending_load = 'example'
            st.rerun()
    
    # Generate question using Model A 3-step pipeline
    if st.session_state.article:
        if st.button("Generate Question (3-Step)", type="primary", use_container_width=True, key="model_gen_btn"):
            # Load inference to pass to generation function
            try:
                # Try to get inference from session state if available
                inference = st.session_state.get('inference')
                qa_data = load_model_generated_questions_mode(st.session_state.article, inference)
            except Exception as e:
                # Fallback without model A ranking
                qa_data = load_model_generated_questions_mode(st.session_state.article)
            
            st.session_state.question = qa_data['question']
            st.session_state.article = qa_data['article']
            st.session_state.options = qa_data['options']
            st.session_state.correct_answer = qa_data['correct_answer']
            st.session_state.generated_questions = qa_data.get('generated_questions', [])
            st.session_state.question_bundles = qa_data.get('question_bundles', [])
            st.session_state.current_question_index = 0
            
            st.success("✅ Full Q&A Generated (Question + Answer + Distractors)!")
            st.info(f"**Pipeline:**\n{qa_data.get('pipeline_description', 'Step 1→2→3')}\n\n{qa_data['question_hint']}")
            st.rerun()
    
    # Question display
    st.markdown("### Generated Question")
    st.text_input(
        "Question:",
        value=st.session_state.question,
        disabled=True,
        key="model_gen_question_display"
    )
    
    if not st.session_state.question:
        st.warning("Click 'Generate Question (3-Step)' to generate full Q&A")
        return

    if st.session_state.generated_questions:
        with st.expander(f"View all {len(st.session_state.generated_questions)} generated questions", expanded=False):
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("Previous", key="model_gen_prev_question"):
                    set_question_index(max(0, st.session_state.current_question_index - 1))
                    st.rerun()
            with nav_col2:
                st.markdown(
                    f"**Active question:** {st.session_state.current_question_index + 1} / {len(st.session_state.generated_questions)}"
                )
            with nav_col3:
                if st.button("Next", key="model_gen_next_question"):
                    set_question_index(min(len(st.session_state.generated_questions) - 1, st.session_state.current_question_index + 1))
                    st.rerun()

            for idx, question_data in enumerate(st.session_state.generated_questions, 1):
                prefix = "▶ " if (idx - 1) == st.session_state.current_question_index else ""
                st.markdown(
                    f"{prefix}{idx}. {question_data['question']}  \n"
                    f"   - Type: {question_data.get('template_type', 'unknown')}  \n"
                    f"   - Source: {question_data.get('source_sentence', '')[:120]}"
                )

    if st.session_state.get('question_bundles'):
        active_bundle = st.session_state.question_bundles[st.session_state.current_question_index]
        st.markdown("### Active Question Bundle")
        st.text_input("Active Question", value=active_bundle['question'], disabled=True, key="model_gen_active_question")
        st.text_input("Generated Answer", value=active_bundle['answer'], disabled=True, key="model_gen_active_answer")
    
    # Display generated answer and distractors (read-only)
    st.markdown("### Generated Answer & Distractors")
    
    correct_option = st.session_state.options[st.session_state.correct_answer]
    distractors_display = [opt for i, opt in enumerate(st.session_state.options) if i != st.session_state.correct_answer]
    
    col_ans, col_dist = st.columns([1, 2])
    
    with col_ans:
        st.text_input(
            "✓ Correct Answer (Generated):",
            value=correct_option,
            disabled=True,
            key="model_gen_answer_display"
        )
    
    with col_dist:
        st.markdown("**Distractors (Generated):**")
        for i, dist in enumerate(distractors_display, 1):
            st.text_input(
                f"Wrong Option {i}:",
                value=dist,
                disabled=True,
                key=f"model_gen_dist_{i}"
            )
    
    st.divider()
    
    # Option to edit if needed
    with st.expander("⚙️ Edit Generated Q&A (Optional)"):
        st.warning("Edit only if the generated Q&A needs improvement")
        
        edited_answer = st.text_input(
            "Edit Answer:",
            value=correct_option,
            key="model_gen_edit_answer"
        )
        
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            edited_dist1 = st.text_input("Edit Distractor 1:", value=distractors_display[0] if len(distractors_display) > 0 else "", key="model_gen_edit_dist1")
        with col_e2:
            edited_dist2 = st.text_input("Edit Distractor 2:", value=distractors_display[1] if len(distractors_display) > 1 else "", key="model_gen_edit_dist2")
        with col_e3:
            edited_dist3 = st.text_input("Edit Distractor 3:", value=distractors_display[2] if len(distractors_display) > 2 else "", key="model_gen_edit_dist3")
        
        if st.button("Update Q&A", key="model_gen_update"):
            st.session_state.options = [edited_answer, edited_dist1, edited_dist2, edited_dist3]
            st.session_state.correct_answer = 0
            st.success("Q&A updated!")
            st.rerun()
    
    # Proceed button
    if st.button("Next to Quiz", type="primary", use_container_width=True, key="model_gen_next"):
        set_screen('quiz_view')
        st.rerun()


# =============================================================================
# MODE 3: User-Provided Questions
# =============================================================================

def render_user_provided_mode():
    """MODE 3: User writes everything."""
    st.info("""
    ### Mode 3: Full Custom Control
    1. **You provide the article**
    2. **You write the question**
    3. **You write answer and options** (3 wrong options)
    4. **Model A verifies** the Q&A
    5. **Model B improves** distractors and hints
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Article")
        article = st.text_area(
            "Paste your article:",
            value=st.session_state.article,
            height=150,
            placeholder="Enter article text...",
            key="user_article"
        )
        st.session_state.article = article
    
    with col2:
        st.markdown("### Quick Load")
        if st.button("Random Article", use_container_width=True, key="user_random"):
            st.session_state.pending_load = 'random'
            st.rerun()
        
        if st.button("Example (Great Wall)", use_container_width=True, key="user_example"):
            st.session_state.pending_load = 'example'
            st.rerun()
    
    st.markdown("### Question")
    question = st.text_input(
        "Write the question:",
        value=st.session_state.question,
        placeholder="What is...?",
        key="user_question"
    )
    st.session_state.question = question
    
    st.markdown("### Answer & Options")
    
    answer = st.text_input(
        "Correct Answer:",
        placeholder="The right answer",
        key="user_answer"
    )
    st.session_state.options[0] = answer
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        opt_a = st.text_input("Wrong Option 1:", key="user_opt_a")
        st.session_state.options[1] = opt_a
    
    with col_b:
        opt_b = st.text_input("Wrong Option 2:", key="user_opt_b")
        st.session_state.options[2] = opt_b
    
    with col_c:
        opt_c = st.text_input("Wrong Option 3:", key="user_opt_c")
        st.session_state.options[3] = opt_c
    
    st.session_state.correct_answer = 0  # User's answer is always correct
    
    # Proceed button
    if st.button("Next to Quiz", type="primary", use_container_width=True, key="user_next"):
        if not all([st.session_state.article, st.session_state.question, st.session_state.options[0]]):
            st.error("Please fill in all fields!")
        else:
            set_screen('quiz_view')
            st.rerun()

"""Analytics Dashboard component (Screen 4)"""

import streamlit as st
from .header import render_screen_title
from .utils import get_option_letter, set_screen, get_active_question_data

def render_analytics_dashboard():
    """Render Screen 4: Analytics Dashboard."""
    render_screen_title(4, "Analytics Dashboard")

    active_question, current_index, total_questions = get_active_question_data()
    if total_questions > 1:
        st.markdown(f"**Question {current_index + 1} of {total_questions}**")
        st.markdown(f"**Active Question:** {active_question['question']}")
    
    inference = st.session_state.get('inference')
    
    if not inference:
        st.error("Inference model not loaded")
        if st.button("Retry Loading Inference", key="btn_retry_inference"):
            try:
                from ui.app import load_models
                st.session_state.inference = load_models()
            except Exception as exc:
                st.error(f"Retry failed: {exc}")
            st.rerun()
        return
    
    # User answer vs correct answer
    render_answer_comparison()
    
    # Model A Predictions
    render_model_a_predictions(inference)
    
    # Model B Distractor Ranking
    render_model_b_predictions(inference)
    
    # Navigation buttons
    render_analytics_buttons()

def render_answer_comparison():
    """Render user answer vs correct answer comparison."""
    active_question, _, _ = get_active_question_data()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Your Answer")
        if st.session_state.user_answer is not None:
            your_answer = st.session_state.options[st.session_state.user_answer]
            st.markdown(f"**Option {get_option_letter(st.session_state.user_answer)}: {your_answer}**")
        else:
            st.warning("No answer selected")
    
    with col2:
        st.markdown("### Correct Answer")
        correct_answer = active_question.get('answer', st.session_state.options[st.session_state.correct_answer])
        st.markdown(f"**Option {get_option_letter(st.session_state.correct_answer)}: {correct_answer}**")
    
    # User answer verification
    if st.session_state.user_answer is not None:
        is_correct = st.session_state.user_answer == st.session_state.correct_answer
        if is_correct:
            st.success("[CORRECT] Your answer matches the expected answer!")
        else:
            st.error("[INCORRECT] Your answer does not match the expected answer.")

def render_model_a_predictions(inference):
    """Render Model A Q&A Verification predictions."""
    active_question, _, _ = get_active_question_data()
    st.markdown("### Model A - Q&A Verification (Ensemble of 10 Models)")
    
    with st.spinner("Running Model A predictions..."):
        qa_result = inference.verify_qa(
            question=active_question.get('question', st.session_state.question),
            answer=active_question.get('answer', st.session_state.options[st.session_state.correct_answer]),
            article=st.session_state.article
        )
    
    if 'error' in qa_result:
        st.error(f"Error in Model A: {qa_result['error']}")
    else:
        # Display ensemble prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            ensemble_pred = qa_result.get('ensemble_prediction', 0)
            st.metric(
                "Ensemble Prediction",
                f"{ensemble_pred:.1%}",
                delta="Valid" if ensemble_pred > 0.5 else "Invalid"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            votes = qa_result.get('votes_for_valid', 0)
            total = qa_result.get('total_models', 10)
            st.metric(
                "Model Votes",
                f"{votes}/{total}",
                delta="Majority" if votes > total/2 else "Minority"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            is_valid = qa_result.get('is_valid_qa', False)
            status = "[VALID]" if is_valid else "[INVALID]"
            st.markdown(f"### Q&A Status: {status}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Individual model predictions
        with st.expander("View Individual Model Predictions"):
            model_preds = qa_result.get('model_predictions', {})
            
            for model_name, pred_data in model_preds.items():
                pred = pred_data.get('pred', '?')
                conf = pred_data.get('confidence', None)
                
                pred_label = "Valid" if pred == 1 else ("Cluster" if isinstance(pred, int) and pred > 1 else "Invalid")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.text(model_name)
                with col2:
                    st.text(pred_label)
                with col3:
                    if conf is not None:
                        st.text(f"{conf:.2%}")

def render_model_b_predictions(inference):
    """Render Model B Distractor Ranking predictions."""
    active_question, _, _ = get_active_question_data()
    st.markdown("### Model B - Distractor Ranking")
    
    wrong_options = [opt for i, opt in enumerate(active_question.get('options', st.session_state.options)) 
                     if i != st.session_state.correct_answer]
    
    distractor_result = inference.generate_quiz_options(
        correct_answer=active_question.get('answer', st.session_state.options[st.session_state.correct_answer]),
        wrong_options=wrong_options,
        question=active_question.get('question', st.session_state.question),
        article=st.session_state.article
    )
    
    if 'error' in distractor_result:
        st.warning(f"Model B: {distractor_result['error']}")
    else:
        st.markdown("**Ranked Distractors:**")
        for dist in distractor_result.get('distractors', []):
            if isinstance(dist, dict):
                st.markdown(f"- {dist.get('text', '')} (Score: {dist.get('score', 0):.3f})")
            else:
                st.markdown(f"- {dist}")

def render_analytics_buttons():
    """Render action buttons for analytics dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Back to Quiz", use_container_width=True, key="btn_analytics_back_quiz"):
            set_screen('quiz_view')
            st.rerun()
    
    with col2:
        if st.button("View Hints", use_container_width=True, key="btn_analytics_hints"):
            set_screen('hint_panel')
            st.rerun()
    
    with col3:
        if st.button("New Quiz", use_container_width=True, key="btn_analytics_new"):
            set_screen('article_input')
            st.session_state.user_answer = None
            st.session_state.hints = []
            st.rerun()
    
    with col4:
        if st.button("Back to Input", use_container_width=True, key="btn_analytics_back_input"):
            set_screen('article_input')
            st.rerun()

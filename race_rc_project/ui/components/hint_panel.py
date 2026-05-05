"""Hint Panel component (Screen 3)"""

import streamlit as st
from .header import render_screen_title
from .utils import set_screen
from .utils import get_active_question_data


def render_hint_panel():
    """Render Screen 3: Hint Panel.

    Hints are revealed one at a time. The user clicks a button to see
    the next hint. This encourages thinking before peeking.
    """
    render_screen_title(3, "Hint Panel")

    active_question, current_index, total_questions = get_active_question_data()
    if total_questions > 1:
        st.markdown(f"**Question {current_index + 1} of {total_questions}**")
    st.markdown(
        f"**Active Question:** {active_question.get('question', st.session_state.get('question', ''))}"
    )

    st.markdown("---")

    # Display hints one-by-one (NOT the correct answer)
    hints = st.session_state.get('hints', [])
    if hints:
        render_hints_progressive(hints)
    else:
        st.info("No hints available. Generate hints from Quiz View.")

    # Navigation buttons
    render_hint_panel_buttons()


def render_hints_progressive(hints):
    """Render hints one at a time with a reveal button."""

    # Track how many hints the user has revealed so far
    if 'hints_revealed' not in st.session_state:
        st.session_state.hints_revealed = 0

    total_hints = len(hints)
    revealed = st.session_state.hints_revealed

    # Header
    st.markdown(f"### 💡 Hints ({min(revealed, total_hints)} of {total_hints} revealed)")

    # Show revealed hints
    for idx in range(min(revealed, total_hints)):
        hint = hints[idx]
        if isinstance(hint, dict):
            hint_text = hint.get('text', str(hint))
            hint_score = hint.get('score', 0)
        else:
            hint_text = str(hint)
            hint_score = 0

        # Use native Streamlit containers for theme-compatible display
        with st.container():
            # Difficulty indicator
            if hint_score >= 0.75:
                difficulty = "🟢 Easy"
            elif hint_score >= 0.5:
                difficulty = "🟡 Medium"
            else:
                difficulty = "🔴 Hard"

            st.markdown(
                f"**Hint {idx + 1}** &nbsp;·&nbsp; {difficulty} &nbsp;·&nbsp; "
                f"Quality: `{hint_score:.0%}`"
            )
            st.info(f"📌 {hint_text}")

    # Reveal next hint button (if more hints remain)
    if revealed < total_hints:
        remaining = total_hints - revealed
        label = (
            "🔓 Reveal First Hint"
            if revealed == 0
            else f"🔓 Reveal Next Hint ({remaining} remaining)"
        )
        if st.button(label, key="btn_reveal_next_hint", use_container_width=True):
            st.session_state.hints_revealed += 1
            st.rerun()
    else:
        st.success("✅ All hints revealed!")


def render_hint_panel_buttons():
    """Render action buttons for hint panel."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "⬅ Back to Quiz",
            use_container_width=True,
            key="btn_hint_back_quiz",
        ):
            st.session_state.hints_revealed = 0
            set_screen('quiz_view')
            st.rerun()

    with col2:
        if st.button(
            "📊 View Analytics",
            use_container_width=True,
            key="btn_hint_analytics",
        ):
            st.session_state.hints_revealed = 0
            set_screen('analytics')
            st.rerun()

    with col3:
        if st.button(
            "⬅ Back to Input",
            use_container_width=True,
            key="btn_hint_back_input",
        ):
            st.session_state.hints_revealed = 0
            set_screen('article_input')
            st.rerun()

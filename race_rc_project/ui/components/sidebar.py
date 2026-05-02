"""Sidebar component for navigation and state display"""

import streamlit as st
import json
from .utils import set_screen

def render_sidebar():
    """Render sidebar navigation and state display."""
    
    with st.sidebar:
        st.markdown("### Navigation")
        
        screen_options = {
            "article_input": "1. Article Input",
            "quiz_view": "2. Quiz View",
            "hint_panel": "3. Hint Panel",
            "analytics": "4. Analytics"
        }
        
        # Get current screen
        current_screen = st.session_state.get('screen', 'article_input')

        # Keep sidebar radio value aligned with current screen before widget creation.
        if st.session_state.get('sidebar_screen_select') != current_screen:
            st.session_state.sidebar_screen_select = current_screen
        
        # Find current index
        screen_list = list(screen_options.keys())
        try:
            current_idx = screen_list.index(current_screen)
        except ValueError:
            current_idx = 0
        
        # Radio selection
        selected_screen = st.radio(
            "Go to:",
            options=screen_list,
            format_func=lambda x: screen_options[x],
            key="sidebar_screen_select"
        )
        
        # Update screen if changed
        if selected_screen != current_screen:
            set_screen(selected_screen)
            st.rerun()
        
        st.divider()
        
        # Display current state
        st.markdown("### Current State")
        with st.expander("Show state"):
            state_display = {
                "screen": st.session_state.get('screen', 'none'),
                "question": (st.session_state.get('question', '')[:50] + "...") 
                           if st.session_state.get('question') else "None",
                "user_answer": st.session_state.get('user_answer'),
                "correct_answer_idx": st.session_state.get('correct_answer'),
                "has_hints": len(st.session_state.get('hints', [])) > 0,
                "has_inference_results": st.session_state.get('inference_results') is not None,
            }
            st.json(state_display)

def render_navigation_buttons(buttons_dict):
    """Render a set of navigation buttons.
    
    Args:
        buttons_dict: Dictionary of {label: action_callback}
    """
    cols = st.columns(len(buttons_dict))
    
    for col, (label, action) in zip(cols, buttons_dict.items()):
        with col:
            if st.button(label, use_container_width=True):
                action()

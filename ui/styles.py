import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .assistant-message {
            background-color: #f3e5f5;
            border-left: 5px solid #9c27b0;
        }
        .sidebar-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin: 0.5rem 0;
        }
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .status-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True) 
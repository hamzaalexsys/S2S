"""
Modern Chat UI Components with Animations and Interactive Features
"""

import streamlit as st
import re
from typing import Dict, List, Optional
from datetime import datetime
import base64
import json

def apply_modern_chat_styles():
    """Apply modern CSS styles for the chat interface"""
    st.markdown("""
    <style>
        /* Main chat container */
        .chat-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Message bubbles */
        .message-bubble {
            animation: slideIn 0.3s ease-out;
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 18px;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            word-wrap: break-word;
        }
        
        .message-bubble:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.15);
        }
        
        /* User message styling */
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
            border-bottom-right-radius: 5px;
        }
        
        /* Assistant message styling */
        .assistant-message {
            background: rgba(255,255,255,0.95);
            color: #333;
            margin-right: 20%;
            border-bottom-left-radius: 5px;
            border-left: 4px solid #667eea;
        }
        
        /* Message header */
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85em;
            opacity: 0.8;
        }
        
        .message-avatar {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            margin-right: 10px;
        }
        
        .user-avatar {
            background: rgba(255,255,255,0.2);
        }
        
        .assistant-avatar {
            background: #667eea;
            color: white;
        }
        
        /* Message content */
        .message-content {
            line-height: 1.6;
            word-wrap: break-word;
        }
        
        .message-content h1, .message-content h2, .message-content h3 {
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        
        .message-content ul, .message-content ol {
            margin-left: 20px;
            margin-bottom: 1em;
        }
        
        .message-content code {
            background: rgba(0,0,0,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        
        .message-content pre {
            background: rgba(0,0,0,0.05);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }
        
        /* Message actions */
        .message-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .message-bubble:hover .message-actions {
            opacity: 1;
        }
        
        .action-btn {
            background: rgba(0,0,0,0.1);
            border: none;
            border-radius: 6px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s ease;
            color: inherit;
        }
        
        .user-message .action-btn {
            background: rgba(255,255,255,0.2);
        }
        
        .action-btn:hover {
            background: rgba(0,0,0,0.2);
            transform: scale(1.05);
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 15px 20px;
            background: rgba(255,255,255,0.9);
            border-radius: 18px;
            margin: 10px 0;
            margin-right: 20%;
            border-left: 4px solid #667eea;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(2) { animation-delay: -0.32s; }
        .typing-dot:nth-child(3) { animation-delay: -0.16s; }
        
        /* Animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }
        
        /* Source citations */
        .source-citations {
            background: rgba(0,0,0,0.05);
            border-left: 4px solid #667eea;
            padding: 10px 15px;
            margin-top: 15px;
            border-radius: 0 8px 8px 0;
        }
        
        .source-item {
            margin: 5px 0;
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        /* Educational mode styling */
        .educational-content {
            background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #4caf50;
        }
        
        .quiz-container {
            background: #fff3e0;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #ff9800;
        }
        
        .practice-container {
            background: #fce4ec;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #e91e63;
        }
        
        /* Input area styling */
        .input-container {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .message-bubble {
                margin-left: 5%;
                margin-right: 5%;
            }
            
            .user-message {
                margin-left: 10%;
            }
            
            .assistant-message {
                margin-right: 10%;
            }
        }
        
        /* Scroll area improvements */
        .chat-messages {
            max-height: 60vh;
            overflow-y: auto;
            padding: 10px;
            scroll-behavior: smooth;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.1);
            border-radius: 3px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(0,0,0,0.3);
            border-radius: 3px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(0,0,0,0.5);
        }
    </style>
    
    <script>
        function copyMessage(messageId) {
            const messageElement = document.querySelector(`[data-message-id="${messageId}"] .message-content`);
            if (messageElement) {
                const text = messageElement.innerText;
                navigator.clipboard.writeText(text).then(() => {
                    // Show temporary feedback
                    const btn = event.target;
                    const originalText = btn.innerText;
                    btn.innerText = '✓ Copied';
                    setTimeout(() => {
                        btn.innerText = originalText;
                    }, 2000);
                });
            }
        }
        
        function speakMessage(messageId) {
            const messageElement = document.querySelector(`[data-message-id="${messageId}"] .message-content`);
            if (messageElement && 'speechSynthesis' in window) {
                const text = messageElement.innerText;
                const utterance = new SpeechSynthesisUtterance(text);
                speechSynthesis.speak(utterance);
            }
        }
        
        function likeMessage(messageId) {
            // Send like feedback to parent
            window.parent.postMessage({
                action: 'like_message', 
                messageId: messageId
            }, '*');
        }
        
        function dislikeMessage(messageId) {
            // Send dislike feedback to parent
            window.parent.postMessage({
                action: 'dislike_message', 
                messageId: messageId
            }, '*');
        }
    </script>
    """, unsafe_allow_html=True)

def render_modern_message(message: Dict, index: int):
    """Render a modern chat message with improved styling"""
    
    role = message['role']
    content = message['content']
    timestamp = message.get('timestamp', '')
    message_id = message.get('id', f"{role}_{index}")
    sources = message.get('sources', [])
    mode = message.get('mode', 'chat')
    model_used = message.get('model_used', '')
    
    # Message container styling
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">
                <span class="user-icon">👤</span>
                <span class="message-time">{timestamp}</span>
            </div>
            <div class="message-content">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # assistant
        # Assistant message with enhanced features
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">
                <span class="assistant-icon">🤖</span>
                <span class="message-info">
                    {model_used} • {mode} • {timestamp}
                </span>
            </div>
            <div class="message-content">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Source citations if available
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for i, source in enumerate(sources, 1):
                    filename = source.get('metadata', {}).get('filename', f'Document {i}')
                    score = source.get('score', 0.0)
                    text_preview = source.get('text', '')[:200] + "..." if len(source.get('text', '')) > 200 else source.get('text', '')
                    
                    st.markdown(f"""
                    **[Source {i}]** {filename} (relevance: {score:.3f})
                    
                    *{text_preview}*
                    """)
        
        # User feedback section
        render_message_feedback(message_id, content)

def render_message_feedback(message_id: str, content: str):
    """Render user feedback buttons for message quality"""
    
    feedback_key_pos = f"feedback_positive_{message_id}"
    feedback_key_neg = f"feedback_negative_{message_id}"
    feedback_reason_key = f"feedback_reason_{message_id}"
    
    # Check if feedback already given
    if st.session_state.get(feedback_key_pos) or st.session_state.get(feedback_key_neg):
        # Show thank you message
        feedback_type = "positive" if st.session_state.get(feedback_key_pos) else "negative"
        st.success(f"✅ Thank you for your {feedback_type} feedback!")
        return
    
    st.markdown("---")
    st.markdown("**Was this response helpful?**")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("👍 Yes", key=f"pos_{message_id}", help="Mark as helpful"):
            st.session_state[feedback_key_pos] = True
            log_feedback(message_id, "positive", content)
            st.rerun()
    
    with col2:
        if st.button("👎 No", key=f"neg_{message_id}", help="Mark as not helpful"):
            st.session_state[feedback_key_neg] = True
            
            # Show feedback reason options
            with col3:
                reason = st.selectbox(
                    "Why wasn't this helpful?",
                    ["", "Incorrect information", "Missing information", "Not relevant", "Poor quality", "Other"],
                    key=feedback_reason_key
                )
                
                if reason:
                    log_feedback(message_id, "negative", content, reason)
                    st.rerun()

def log_feedback(message_id: str, feedback_type: str, content: str, reason: str = None):
    """Log user feedback for improving the system"""
    
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Create feedback log entry
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'message_id': message_id,
        'feedback_type': feedback_type,
        'content_length': len(content),
        'content_preview': content[:100] + "..." if len(content) > 100 else content,
        'reason': reason,
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    # Append to feedback log file
    feedback_file = Path("logs/user_feedback.jsonl")
    feedback_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        print(f"📝 Feedback logged: {feedback_type} for message {message_id}")
        
        # Also store in session state for analytics
        if 'feedback_log' not in st.session_state:
            st.session_state.feedback_log = []
        st.session_state.feedback_log.append(feedback_entry)
        
    except Exception as e:
        print(f"❌ Failed to log feedback: {e}")

def get_feedback_analytics():
    """Get feedback analytics for system improvement"""
    
    feedback_log = st.session_state.get('feedback_log', [])
    
    if not feedback_log:
        return {
            'total_feedback': 0,
            'positive_count': 0,
            'negative_count': 0,
            'satisfaction_rate': 0.0,
            'common_issues': []
        }
    
    positive_count = sum(1 for entry in feedback_log if entry['feedback_type'] == 'positive')
    negative_count = sum(1 for entry in feedback_log if entry['feedback_type'] == 'negative')
    total_feedback = len(feedback_log)
    
    # Get common negative feedback reasons
    negative_reasons = [entry.get('reason') for entry in feedback_log 
                       if entry['feedback_type'] == 'negative' and entry.get('reason')]
    
    from collections import Counter
    common_issues = Counter(negative_reasons).most_common(3)
    
    return {
        'total_feedback': total_feedback,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'satisfaction_rate': positive_count / total_feedback if total_feedback > 0 else 0.0,
        'common_issues': common_issues
    }

def render_feedback_buttons(message_id: str, role: str) -> str:
    """Render feedback buttons for assistant messages"""
    if role != "assistant":
        return ""
    
    return f'''
    <button class="action-btn" onclick="likeMessage('{message_id}')" title="Helpful response">
        👍 Like
    </button>
    <button class="action-btn" onclick="dislikeMessage('{message_id}')" title="Not helpful">
        👎 Dislike
    </button>
    '''

def process_message_content(content: str, role: str) -> str:
    """Process message content for better display"""
    
    # Convert markdown-style formatting
    content = convert_markdown_to_html(content)
    
    # Handle code blocks
    content = process_code_blocks(content)
    
    # Handle lists
    content = process_lists(content)
    
    # Handle educational formatting
    content = process_educational_formatting(content)
    
    return content

def convert_markdown_to_html(text: str) -> str:
    """Convert basic markdown to HTML"""
    
    # Headers
    text = re.sub(r'^### (.*$)', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*$)', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*$)', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    
    # Line breaks
    text = text.replace('\n', '<br>')
    
    return text

def process_code_blocks(text: str) -> str:
    """Process code blocks for better display"""
    
    # Multi-line code blocks
    pattern = r'```(\w+)?\n(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        return f'<pre><code class="language-{language}">{code}</code></pre>'
    
    text = re.sub(pattern, replace_code_block, text, flags=re.DOTALL)
    
    return text

def process_lists(text: str) -> str:
    """Process lists for better display"""
    
    # Unordered lists
    lines = text.split('<br>')
    in_list = False
    result = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('• ') or stripped.startswith('- '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append('</ul>')
    
    return '<br>'.join(result)

def process_educational_formatting(text: str) -> str:
    """Process educational content formatting"""
    
    # Quiz questions
    text = re.sub(
        r'\*\*Question\*\*:\s*(.*?)<br>',
        r'<div class="quiz-question"><strong>📝 Question:</strong> \1</div><br>',
        text
    )
    
    # Quiz options
    text = re.sub(
        r'([A-D])\)\s*(.*?)<br>',
        r'<div class="quiz-option"><strong>\1)</strong> \2</div><br>',
        text
    )
    
    # Explanations
    text = re.sub(
        r'\*\*Explanation\*\*:\s*(.*?)<br>',
        r'<div class="explanation"><strong>💡 Explanation:</strong> \1</div><br>',
        text
    )
    
    # Key concepts
    text = re.sub(
        r'\*\*Key Concepts\*\*:\s*(.*?)<br>',
        r'<div class="key-concepts"><strong>🔑 Key Concepts:</strong> \1</div><br>',
        text
    )
    
    return text

def render_source_citations(sources: List[Dict]):
    """Render source citations in a nice format"""
    
    if not sources:
        return
    
    citations_html = '<div class="source-citations"><strong>📚 Sources:</strong><br>'
    
    for i, source in enumerate(sources, 1):
        filename = source.get('metadata', {}).get('filename', f'Source {i}')
        score = source.get('score', 0.0)
        
        citations_html += f'''
        <div class="source-item">
            <strong>[{i}]</strong> {filename} 
            <span style="opacity: 0.7;">(relevance: {score:.2f})</span>
        </div>
        '''
    
    citations_html += '</div>'
    
    st.markdown(citations_html, unsafe_allow_html=True)

def render_typing_indicator():
    """Render animated typing indicator"""
    
    st.markdown('''
    <div class="typing-indicator">
        <span>🤖 Assistant is typing</span>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    ''', unsafe_allow_html=True)

def render_chat_input_area(placeholder: str = "Type your message...", 
                          show_voice: bool = True, 
                          show_attachments: bool = True):
    """Render modern chat input area"""
    
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Input controls
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder=placeholder,
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if show_voice:
            voice_button = st.button("🎤", help="Voice input", key="voice_btn")
        else:
            voice_button = False
    
    with col3:
        if show_attachments:
            attachment_button = st.button("📎", help="Attach files", key="attach_btn")
        else:
            attachment_button = False
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'text': user_input,
        'voice': voice_button,
        'attachment': attachment_button
    }

def render_message_suggestions(suggestions: List[str]):
    """Render quick reply suggestions"""
    
    if not suggestions:
        return None
    
    st.markdown("**💡 Quick replies:**")
    
    cols = st.columns(min(len(suggestions), 3))
    
    for i, suggestion in enumerate(suggestions):
        col_index = i % 3
        with cols[col_index]:
            if st.button(f"💬 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                return suggestion
    
    return None

def render_educational_mode_selector():
    """Render educational mode selector"""
    
    st.markdown("**🎓 Educational Mode:**")
    
    modes = {
        'chat': '💬 Chat',
        'explain': '📖 Explain',
        'quiz': '❓ Quiz',
        'practice': '🎯 Practice',
        'summarize': '📝 Summarize'
    }
    
    selected_mode = st.selectbox(
        "Select mode",
        options=list(modes.keys()),
        format_func=lambda x: modes[x],
        key="educational_mode",
        label_visibility="collapsed"
    )
    
    return selected_mode

def render_chat_stats(chat_history: List[Dict]):
    """Render chat statistics"""
    
    if not chat_history:
        return
    
    total_messages = len(chat_history)
    user_messages = len([m for m in chat_history if m['role'] == 'user'])
    assistant_messages = len([m for m in chat_history if m['role'] == 'assistant'])
    
    total_words = sum(
        len(m['content'].split()) 
        for m in chat_history
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", total_messages)
    
    with col2:
        st.metric("Your Messages", user_messages)
    
    with col3:
        st.metric("AI Responses", assistant_messages)
    
    with col4:
        st.metric("Total Words", f"{total_words:,}")

def export_chat_conversation(chat_history: List[Dict], format: str = 'markdown'):
    """Export chat conversation in various formats"""
    
    if not chat_history:
        return None
    
    if format == 'markdown':
        content = "# Chat Conversation\n\n"
        content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for message in chat_history:
            role = "**User**" if message['role'] == 'user' else "**Assistant**"
            timestamp = message.get('timestamp', '')
            content += f"{role} ({timestamp}):\n{message['content']}\n\n---\n\n"
        
        return content.encode()
    
    elif format == 'json':
        return json.dumps(chat_history, indent=2).encode()
    
    elif format == 'html':
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chat Conversation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .message {{ margin: 20px 0; padding: 15px; border-radius: 10px; }}
                .user {{ background-color: #e3f2fd; }}
                .assistant {{ background-color: #f1f8e9; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Chat Conversation</h1>
            <p>Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for message in chat_history:
            role_class = message['role']
            role_name = "You" if message['role'] == 'user' else "Assistant"
            timestamp = message.get('timestamp', '')
            
            html_content += f"""
            <div class="message {role_class}">
                <strong>{role_name}</strong> 
                <span class="timestamp">({timestamp})</span>
                <p>{message['content'].replace('\n', '<br>')}</p>
            </div>
            """
        
        html_content += "</body></html>"
        return html_content.encode()
    
    return None 
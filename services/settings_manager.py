"""
Settings Manager with Profile Support and Persistence
"""

import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

class SettingsManager:
    def __init__(self, settings_dir: str = ".streamlit_assistant"):
        # Use user's home directory for cross-platform compatibility
        self.settings_dir = Path.home() / settings_dir
        self.settings_dir.mkdir(exist_ok=True)
        
        # Sub-directories
        self.profiles_dir = self.settings_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.cache_dir = self.settings_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.exports_dir = self.settings_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        
        # Default settings
        self.default_settings = {
            'selected_model': 'llama3.2:1b',
            'whisper_model_size': 'base',
            'enable_speech_input': True,
            'enable_speech_output': True,
            'enable_educational_mode': False,
            'chat_mode': 'chat',
            'theme': 'modern',
            'voice_speed': 1.0,
            'voice_pitch': 1.0,
            'auto_play_responses': False,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'search_method': 'hybrid',
            'max_context_length': 4000,
            'temperature': 0.7,
            'show_source_citations': True,
            'auto_save_conversations': True,
            'conversation_export_format': 'markdown',
            'ui_animations': True,
            'compact_mode': False,
            'developer_mode': False
        }
    
    def save_settings(self, settings: Dict, profile_name: str = "default") -> bool:
        """Save current settings to profile"""
        try:
            profile_file = self.profiles_dir / f"{profile_name}.json"
            
            # Add metadata
            settings_with_metadata = {
                'settings': settings,
                'metadata': {
                    'profile_name': profile_name,
                    'created_at': datetime.now().isoformat(),
                    'last_modified': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            with open(profile_file, 'w') as f:
                json.dump(settings_with_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
            return False
    
    def load_settings(self, profile_name: str = "default") -> Optional[Dict]:
        """Load settings from profile"""
        try:
            profile_file = self.profiles_dir / f"{profile_name}.json"
            
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both old and new format
                if 'settings' in data:
                    return data['settings']
                else:
                    return data  # Old format
            
            return self.default_settings.copy()
            
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
            return self.default_settings.copy()
    
    def list_profiles(self) -> List[str]:
        """List available profiles"""
        try:
            profiles = []
            for file in self.profiles_dir.glob("*.json"):
                profiles.append(file.stem)
            return sorted(profiles)
        except Exception:
            return []
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile"""
        try:
            if profile_name == "default":
                return False  # Can't delete default profile
            
            profile_file = self.profiles_dir / f"{profile_name}.json"
            if profile_file.exists():
                profile_file.unlink()
                return True
            return False
            
        except Exception as e:
            st.error(f"Error deleting profile: {str(e)}")
            return False
    
    def export_profile(self, profile_name: str) -> Optional[str]:
        """Export profile for sharing"""
        try:
            profile_file = self.profiles_dir / f"{profile_name}.json"
            
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            st.error(f"Error exporting profile: {str(e)}")
            return None
    
    def import_profile(self, profile_data: str, profile_name: str) -> bool:
        """Import profile from JSON data"""
        try:
            data = json.loads(profile_data)
            
            # Validate the data structure
            if 'settings' in data or any(key in data for key in self.default_settings.keys()):
                return self.save_settings(data.get('settings', data), profile_name)
            else:
                st.error("Invalid profile format")
                return False
                
        except Exception as e:
            st.error(f"Error importing profile: {str(e)}")
            return False
    
    def get_profile_metadata(self, profile_name: str) -> Optional[Dict]:
        """Get profile metadata"""
        try:
            profile_file = self.profiles_dir / f"{profile_name}.json"
            
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                return data.get('metadata', {})
            
            return None
            
        except Exception:
            return None
    
    def backup_all_profiles(self) -> str:
        """Create a backup of all profiles"""
        try:
            backup_data = {
                'backup_date': datetime.now().isoformat(),
                'profiles': {}
            }
            
            for profile_name in self.list_profiles():
                profile_file = self.profiles_dir / f"{profile_name}.json"
                with open(profile_file, 'r') as f:
                    backup_data['profiles'][profile_name] = json.load(f)
            
            backup_filename = f"profiles_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.exports_dir / backup_filename
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return str(backup_path)
            
        except Exception as e:
            st.error(f"Error creating backup: {str(e)}")
            return ""
    
    def restore_from_backup(self, backup_data: str) -> bool:
        """Restore profiles from backup"""
        try:
            data = json.loads(backup_data)
            
            if 'profiles' not in data:
                st.error("Invalid backup format")
                return False
            
            success_count = 0
            for profile_name, profile_data in data['profiles'].items():
                if self.save_settings(
                    profile_data.get('settings', profile_data), 
                    profile_name
                ):
                    success_count += 1
            
            st.success(f"Restored {success_count} profiles")
            return success_count > 0
            
        except Exception as e:
            st.error(f"Error restoring backup: {str(e)}")
            return False
    
    def get_cache_size(self) -> str:
        """Get total cache size"""
        try:
            total_size = 0
            for file in self.cache_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            
            # Convert to human readable
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
            
        except Exception:
            return "Unknown"
    
    def clear_cache(self) -> bool:
        """Clear all cache files"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
            return False

def render_profile_manager(session_manager):
    """Render profile management interface without nested expanders"""
    st.subheader("👤 Profile Management")
    
    settings_manager = SettingsManager()
    profiles = settings_manager.list_profiles()
    
    if not profiles:
        st.info("No profiles found. Create your first profile below.")
        profiles = ["default"]
    
    # Profile selection
    current_profile = session_manager.get('current_profile', 'default')
    selected_profile = st.selectbox(
        "Select Profile",
        profiles,
        index=profiles.index(current_profile) if current_profile in profiles else 0,
        key="profile_selector"
    )
    
    if selected_profile != current_profile:
        # Load selected profile
        profile_settings = settings_manager.load_settings(selected_profile)
        if profile_settings:
            for key, value in profile_settings.items():
                session_manager.set(key, value)
            session_manager.set('current_profile', selected_profile)
            st.success(f"Loaded profile: {selected_profile}")
            st.rerun()
    
    # Profile actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current Settings", use_container_width=True):
            current_settings = get_current_settings(session_manager)
            if settings_manager.save_settings(current_settings, selected_profile):
                st.success(f"Settings saved to {selected_profile}")
            else:
                st.error("Failed to save settings")
    
    with col2:
        if selected_profile != "default":
            if st.button("🗑️ Delete Profile", use_container_width=True):
                if st.session_state.get(f'confirm_delete_{selected_profile}', False):
                    if settings_manager.delete_profile(selected_profile):
                        st.success(f"Profile '{selected_profile}' deleted")
                        session_manager.set('current_profile', 'default')
                        st.rerun()
                    else:
                        st.error("Failed to delete profile")
                else:
                    st.session_state[f'confirm_delete_{selected_profile}'] = True
                    st.warning("Click again to confirm deletion")
    
    # Create new profile section
    st.markdown("**➕ Create New Profile**")
    new_profile_name = st.text_input(
        "Profile Name", 
        placeholder="Enter profile name...",
        key="new_profile_name"
    )
    
    if st.button("Create Profile") and new_profile_name:
        if new_profile_name not in profiles:
            current_settings = get_current_settings(session_manager)
            if settings_manager.save_settings(current_settings, new_profile_name):
                st.success(f"Profile '{new_profile_name}' created")
                st.rerun()
            else:
                st.error("Failed to create profile")
        else:
            st.error("Profile name already exists")
    
    # Import/Export section
    st.markdown("**📥📤 Import/Export Profiles**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Export Profile*")
        export_profile = st.selectbox(
            "Select profile to export:",
            profiles,
            key="export_profile_select"
        )
        
        if st.button("📤 Export"):
            profile_data = settings_manager.export_profile(export_profile)
            if profile_data:
                st.download_button(
                    label="💾 Download Profile JSON",
                    data=profile_data,
                    file_name=f"{export_profile}_profile.json",
                    mime="application/json"
                )
    
    with col2:
        st.markdown("*Import Profile*")
        uploaded_profile = st.file_uploader(
            "Choose profile JSON file",
            type=['json'],
            key="profile_upload"
        )
        
        if uploaded_profile:
            import_name = st.text_input(
                "Profile name:",
                value=uploaded_profile.name.replace('.json', '').replace('_profile', ''),
                key="import_profile_name"
            )
            
            if st.button("📥 Import"):
                profile_data = uploaded_profile.read().decode()
                if settings_manager.import_profile(profile_data, import_name):
                    st.success(f"Profile '{import_name}' imported successfully")
                    st.rerun()
    
    # Show profile metadata
    if selected_profile:
        metadata = settings_manager.get_profile_metadata(selected_profile)
        if metadata:
            st.markdown("**ℹ️ Profile Information**")
            st.write(f"**Created:** {metadata.get('created_at', 'Unknown')}")
            st.write(f"**Last Modified:** {metadata.get('last_modified', 'Unknown')}")
            st.write(f"**Version:** {metadata.get('version', 'Unknown')}")

def get_current_settings(session_manager) -> Dict:
    """Get current settings from session manager"""
    settings_manager = SettingsManager()
    current_settings = {}
    
    for key in settings_manager.default_settings.keys():
        current_settings[key] = session_manager.get(key, settings_manager.default_settings[key])
    
    return current_settings

def render_advanced_settings(session_manager):
    """Render advanced settings panel without nested expanders"""
    st.subheader("⚙️ Advanced Settings")
    
    settings_manager = SettingsManager()
    
    # Performance settings
    st.markdown("**🚀 Performance**")
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.slider(
            "Document Chunk Size",
            min_value=500,
            max_value=2000,
            value=session_manager.get('chunk_size', 1000),
            help="Size of text chunks for processing"
        )
        session_manager.set('chunk_size', chunk_size)
        
        max_context = st.slider(
            "Max Context Length",
            min_value=2000,
            max_value=8000,
            value=session_manager.get('max_context_length', 4000),
            help="Maximum context length for AI responses"
        )
        session_manager.set('max_context_length', max_context)
    
    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=500,
            value=session_manager.get('chunk_overlap', 200),
            help="Overlap between text chunks"
        )
        session_manager.set('chunk_overlap', chunk_overlap)
        
        temperature = st.slider(
            "AI Temperature",
            min_value=0.0,
            max_value=2.0,
            value=session_manager.get('temperature', 0.7),
            step=0.1,
            help="Controls randomness in AI responses"
        )
        session_manager.set('temperature', temperature)
    
    # Search settings
    st.markdown("**🔍 Search & Retrieval**")
    search_method = st.selectbox(
        "Search Method",
        ['hybrid', 'dense', 'keyword'],
        index=['hybrid', 'dense', 'keyword'].index(
            session_manager.get('search_method', 'hybrid')
        ),
        help="Method used for document retrieval"
    )
    session_manager.set('search_method', search_method)
    
    show_citations = st.checkbox(
        "Show Source Citations",
        value=session_manager.get('show_source_citations', True),
        help="Display source citations in responses"
    )
    session_manager.set('show_source_citations', show_citations)
    
    # UI settings
    st.markdown("**🎨 User Interface**")
    col1, col2 = st.columns(2)
    
    with col1:
        ui_animations = st.checkbox(
            "Enable Animations",
            value=session_manager.get('ui_animations', True),
            help="Enable UI animations and transitions"
        )
        session_manager.set('ui_animations', ui_animations)
        
        auto_save = st.checkbox(
            "Auto-save Conversations",
            value=session_manager.get('auto_save_conversations', True),
            help="Automatically save conversation history"
        )
        session_manager.set('auto_save_conversations', auto_save)
    
    with col2:
        compact_mode = st.checkbox(
            "Compact Mode",
            value=session_manager.get('compact_mode', False),
            help="Use compact UI layout"
        )
        session_manager.set('compact_mode', compact_mode)
        
        export_format = st.selectbox(
            "Export Format",
            ['markdown', 'json', 'html'],
            index=['markdown', 'json', 'html'].index(
                session_manager.get('conversation_export_format', 'markdown')
            ),
            help="Default format for conversation exports"
        )
        session_manager.set('conversation_export_format', export_format)
    
    # Developer settings
    st.markdown("**🔧 Developer**")
    developer_mode = st.checkbox(
        "Developer Mode",
        value=session_manager.get('developer_mode', False),
        help="Enable developer features and debug info"
    )
    session_manager.set('developer_mode', developer_mode)
    
    # Cache management
    st.markdown("**🗂️ Cache Management**")
    cache_size = settings_manager.get_cache_size()
    st.write(f"**Cache Size:** {cache_size}")
    
    if st.button("🗑️ Clear Cache"):
        if settings_manager.clear_cache():
            st.success("Cache cleared successfully")
        else:
            st.error("Failed to clear cache")
    
    # Backup/Restore
    st.markdown("**💾 Backup & Restore**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 Create Backup"):
            backup_path = settings_manager.backup_all_profiles()
            if backup_path:
                with open(backup_path, 'r') as f:
                    backup_data = f.read()
                
                st.download_button(
                    label="💾 Download Backup",
                    data=backup_data,
                    file_name=f"assistant_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        backup_file = st.file_uploader(
            "Restore from backup",
            type=['json'],
            key="backup_restore"
        )
        
        if backup_file and st.button("🔄 Restore"):
            backup_data = backup_file.read().decode()
            if settings_manager.restore_from_backup(backup_data):
                st.success("Backup restored successfully")
                st.rerun() 
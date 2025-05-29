"""
System Analyzer for Smart Model Management
"""

import psutil
import platform
import subprocess
import json
from typing import Dict, List, Optional
import streamlit as st

class SystemAnalyzer:
    def __init__(self):
        self.model_requirements = {
            'llama3.2:1b': {
                'ram': 2, 
                'description': 'Fastest, good for chat',
                'size': '1.3GB',
                'use_cases': ['chat', 'simple_qa'],
                'quality': 'good',
                'speed': 'fast'
            },
            'llama3.2': {
                'ram': 4, 
                'description': 'Balanced performance',
                'size': '3.8GB',
                'use_cases': ['chat', 'qa', 'analysis'],
                'quality': 'better',
                'speed': 'moderate'
            },
            'mistral:7b': {
                'ram': 8, 
                'description': 'High quality responses',
                'size': '4.1GB',
                'use_cases': ['chat', 'qa', 'reasoning'],
                'quality': 'better',
                'speed': 'moderate'
            },
            'mixtral:8x7b': {
                'ram': 48, 
                'description': 'Best quality, requires powerful hardware',
                'size': '26GB',
                'use_cases': ['complex_reasoning', 'coding', 'analysis'],
                'quality': 'excellent',
                'speed': 'slow'
            },
            'phi3:mini': {
                'ram': 2, 
                'description': 'Efficient, good reasoning',
                'size': '2.3GB',
                'use_cases': ['chat', 'reasoning', 'coding'],
                'quality': 'good',
                'speed': 'fast'
            },
            'gemma2:2b': {
                'ram': 3, 
                'description': 'Google model, fast',
                'size': '1.6GB',
                'use_cases': ['chat', 'qa'],
                'quality': 'good',
                'speed': 'fast'
            },
            'qwen2.5:3b': {
                'ram': 4, 
                'description': 'Good multilingual support',
                'size': '1.9GB',
                'use_cases': ['chat', 'qa', 'multilingual'],
                'quality': 'good',
                'speed': 'moderate'
            }
        }
    
    def get_system_info(self) -> Dict:
        """Analyze system capabilities"""
        info = {
            'os': platform.system(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'cpu': {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'model': platform.processor(),
                'frequency': self._get_cpu_frequency()
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 1),
                'used_gb': round(psutil.virtual_memory().used / (1024**3), 1),
                'percentage': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': round(psutil.disk_usage('/').total / (1024**3), 1),
                'free_gb': round(psutil.disk_usage('/').free / (1024**3), 1),
                'used_gb': round(psutil.disk_usage('/').used / (1024**3), 1)
            },
            'gpu': self._detect_gpu()
        }
        return info
    
    def _get_cpu_frequency(self) -> Optional[float]:
        """Get CPU frequency if available"""
        try:
            freq = psutil.cpu_freq()
            return freq.current if freq else None
        except:
            return None
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU availability"""
        gpu_info = {'available': False, 'type': None, 'memory_gb': 0, 'name': 'None'}
        
        try:
            # Try NVIDIA first
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total', 
                '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = lines[0].split(', ')
                if len(gpu_data) >= 2:
                    name = gpu_data[0]
                    memory_str = gpu_data[1]
                    memory_mb = int(memory_str.replace(' MiB', ''))
                    
                    gpu_info = {
                        'available': True,
                        'type': 'NVIDIA',
                        'name': name,
                        'memory_gb': round(memory_mb / 1024, 1)
                    }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Try AMD/Intel GPU detection
            try:
                # This is a simplified detection - in practice, you might use specific libraries
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                if 'VGA' in result.stdout or 'Display' in result.stdout:
                    # Basic GPU detection without memory info
                    gpu_info = {
                        'available': True,
                        'type': 'Integrated/AMD',
                        'name': 'Detected GPU',
                        'memory_gb': 0  # Unknown
                    }
            except:
                pass
        
        return gpu_info
    
    def recommend_models(self) -> Dict:
        """Recommend models based on system capabilities"""
        info = self.get_system_info()
        recommendations = []
        
        available_ram = info['memory']['available_gb']
        total_ram = info['memory']['total_gb']
        
        # Use 70% of available RAM as safe threshold
        safe_ram = available_ram * 0.7
        
        for model, reqs in self.model_requirements.items():
            is_suitable = reqs['ram'] <= safe_ram
            
            # Performance rating based on system specs
            performance_rating = self._calculate_performance_rating(
                model, info, is_suitable
            )
            
            recommendations.append({
                'model': model,
                'description': reqs['description'],
                'ram_required': reqs['ram'],
                'size': reqs['size'],
                'suitable': is_suitable,
                'performance_rating': performance_rating,
                'quality': reqs['quality'],
                'speed': reqs['speed'],
                'use_cases': reqs['use_cases']
            })
        
        # Sort by suitability and performance rating
        recommendations.sort(
            key=lambda x: (x['suitable'], x['performance_rating']), 
            reverse=True
        )
        
        return {
            'system_info': info,
            'recommendations': recommendations,
            'best_model': recommendations[0]['model'] if recommendations else None
        }
    
    def _calculate_performance_rating(self, model: str, system_info: Dict, 
                                     is_suitable: bool) -> float:
        """Calculate performance rating for a model on this system"""
        if not is_suitable:
            return 0.0
        
        reqs = self.model_requirements[model]
        
        # Base rating
        rating = 0.5
        
        # RAM score (0-3 points)
        available_ram = system_info['memory']['available_gb']
        if available_ram > reqs['ram'] * 2:
            rating += 3.0
        elif available_ram > reqs['ram'] * 1.5:
            rating += 2.0
        elif available_ram > reqs['ram']:
            rating += 1.0
        
        # CPU score (0-2 points)
        cpu_threads = system_info['cpu']['threads']
        if cpu_threads >= 8:
            rating += 2.0
        elif cpu_threads >= 4:
            rating += 1.0
        
        # GPU bonus (0-1 point)
        if system_info['gpu']['available']:
            rating += 1.0
        
        # Quality adjustment
        quality_multiplier = {
            'good': 1.0,
            'better': 1.2,
            'excellent': 1.5
        }
        rating *= quality_multiplier.get(reqs['quality'], 1.0)
        
        return min(rating, 10.0)  # Cap at 10
    
    def render_system_info_ui(self):
        """Render system information in Streamlit UI"""
        analysis = self.recommend_models()
        system_info = analysis['system_info']
        
        st.subheader("🖥️ System Information")
        
        # System overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "CPU Cores", 
                f"{system_info['cpu']['cores']} ({system_info['cpu']['threads']} threads)"
            )
        
        with col2:
            st.metric(
                "Available RAM", 
                f"{system_info['memory']['available_gb']:.1f} GB",
                f"of {system_info['memory']['total_gb']:.1f} GB total"
            )
        
        with col3:
            gpu_name = system_info['gpu']['name']
            if system_info['gpu']['available']:
                gpu_display = f"✅ {gpu_name}"
                if system_info['gpu']['memory_gb'] > 0:
                    gpu_display += f" ({system_info['gpu']['memory_gb']:.1f} GB)"
            else:
                gpu_display = "❌ No GPU detected"
            
            st.metric("GPU", gpu_display)
        
        # Memory usage details (without nested expander)
        st.markdown("**📊 Detailed System Status**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Memory usage
            memory = system_info['memory']
            st.write("**Memory Usage**")
            st.progress(memory['percentage'] / 100)
            st.write(f"Used: {memory['used_gb']:.1f} GB / {memory['total_gb']:.1f} GB")
        
        with col2:
            # Disk usage
            disk = system_info['disk']
            disk_percentage = (disk['used_gb'] / disk['total_gb']) * 100
            st.write("**Disk Usage**")
            st.progress(disk_percentage / 100)
            st.write(f"Used: {disk['used_gb']:.1f} GB / {disk['total_gb']:.1f} GB")
        
        return analysis
    
    def render_model_recommendations(self, analysis: Dict):
        """Render model recommendations UI"""
        st.subheader("🤖 Recommended Models")
        
        recommendations = analysis['recommendations']
        
        # Quick selection buttons
        suitable_models = [r for r in recommendations if r['suitable']]
        
        if suitable_models:
            st.write("**Quick Selection:**")
            cols = st.columns(min(len(suitable_models), 4))
            
            for i, model in enumerate(suitable_models[:4]):
                with cols[i]:
                    if st.button(
                        f"{model['model']}\n({model['quality']} quality)",
                        key=f"quick_select_{model['model']}"
                    ):
                        return model['model']
        
        # Simplified recommendations for sidebar (no nested expanders)
        st.write("**Available Models:**")
        
        for model in recommendations:
            # Show model status and basic info
            status_icon = '✅' if model['suitable'] else '❌'
            rating = model['performance_rating']
            
            st.markdown(f"""
            **{status_icon} {model['model']}** ({model['quality']})
            - RAM: {model['ram_required']} GB | Rating: {rating:.1f}/10
            - {model['description']}
            """)
            
            if model['suitable']:
                if st.button(f"Select {model['model']}", 
                           key=f"select_{model['model']}", 
                           use_container_width=True):
                    return model['model']
            else:
                st.caption("❌ Requires more RAM than available")
        
        return None
    
    def render_model_recommendations_full(self, analysis: Dict):
        """Render full model recommendations UI with expanders (for main area)"""
        st.subheader("🤖 Recommended Models")
        
        recommendations = analysis['recommendations']
        
        # Quick selection buttons
        suitable_models = [r for r in recommendations if r['suitable']]
        
        if suitable_models:
            st.write("**Quick Selection:**")
            cols = st.columns(min(len(suitable_models), 4))
            
            for i, model in enumerate(suitable_models[:4]):
                with cols[i]:
                    if st.button(
                        f"{model['model']}\n({model['quality']} quality)",
                        key=f"full_quick_select_{model['model']}"
                    ):
                        return model['model']
        
        # Detailed recommendations with expanders
        st.write("**Detailed Recommendations:**")
        
        for model in recommendations:
            with st.expander(
                f"{'✅' if model['suitable'] else '❌'} {model['model']} "
                f"- {model['description']}"
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**RAM Required:** {model['ram_required']} GB")
                    st.write(f"**Model Size:** {model['size']}")
                    st.write(f"**Quality:** {model['quality']}")
                
                with col2:
                    st.write(f"**Speed:** {model['speed']}")
                    st.write(f"**Performance Rating:** {model['performance_rating']:.1f}/10")
                    if model['suitable']:
                        st.success("✅ Compatible with your system")
                    else:
                        st.error("❌ Requires more RAM")
                
                with col3:
                    st.write("**Use Cases:**")
                    for use_case in model['use_cases']:
                        st.write(f"• {use_case}")
                
                if model['suitable']:
                    if st.button(f"Select {model['model']}", key=f"full_select_{model['model']}"):
                        return model['model']
        
        return None
    
    def check_ollama_status(self) -> Dict:
        """Check if Ollama is running and available"""
        try:
            print("🔍 Checking Ollama status...")
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=10  # Increased timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Ollama command successful. Output:\n{result.stdout}")
                # Parse installed models - improved parsing
                lines = result.stdout.strip().split('\n')
                installed_models = []
                
                # Skip header line (NAME, ID, SIZE, MODIFIED)
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith('-') and not line.startswith('NAME'):  # Skip separator lines and header
                        # Split by whitespace and take first part (model name)
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            # Clean up the model name (remove tags like :latest, :7b, etc.)
                            base_name = model_name.split(':')[0]
                            
                            # Add both full name and base name
                            if model_name not in installed_models:
                                installed_models.append(model_name)
                                print(f"   Found model: {model_name}")
                            if base_name != model_name and base_name not in installed_models:
                                installed_models.append(base_name)
                                print(f"   Found base model: {base_name}")
                
                # Also try to get models via 'ollama ps' for running models
                try:
                    ps_result = subprocess.run(
                        ['ollama', 'ps'], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if ps_result.returncode == 0:
                        print(f"📋 Running models check successful. Output:\n{ps_result.stdout}")
                        ps_lines = ps_result.stdout.strip().split('\n')
                        for line in ps_lines[1:]:  # Skip header
                            line = line.strip()
                            if line and not line.startswith('NAME'):
                                parts = line.split()
                                if parts:
                                    model_name = parts[0]
                                    base_name = model_name.split(':')[0]
                                    if model_name not in installed_models:
                                        installed_models.append(model_name)
                                        print(f"   Found running model: {model_name}")
                                    if base_name != model_name and base_name not in installed_models:
                                        installed_models.append(base_name)
                                        print(f"   Found running base model: {base_name}")
                    else:
                        print(f"⚠️ 'ollama ps' failed: {ps_result.stderr}")
                except Exception as e:
                    print(f"⚠️ Error checking running models: {str(e)}")
                    pass  # ps command is optional
                
                print(f"🔍 Final detected models: {installed_models}")
                return {
                    'running': True,
                    'installed_models': installed_models,
                    'error': None
                }
            else:
                error_msg = f"Ollama command failed: {result.stderr.strip()}"
                print(f"❌ {error_msg}")
                return {
                    'running': False,
                    'installed_models': [],
                    'error': error_msg
                }
        
        except subprocess.TimeoutExpired:
            error_msg = "Ollama command timed out - service may be slow"
            print(f"⏰ {error_msg}")
            return {
                'running': False,
                'installed_models': [],
                'error': error_msg
            }
        except FileNotFoundError:
            error_msg = "Ollama not found - please install Ollama first"
            print(f"❌ {error_msg}")
            return {
                'running': False,
                'installed_models': [],
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Error checking Ollama: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'running': False,
                'installed_models': [],
                'error': error_msg
            } 
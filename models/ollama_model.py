import requests
import subprocess
from typing import Dict, List, Optional, Any
from langchain_community.llms import Ollama

class OllamaModel:
    def __init__(self, config):
        self.config = config
        self.base_url = config.base_url
    
    def check_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'status': 'running',
                    'models': [model['name'] for model in models]
                }
            else:
                return {'status': 'not_running', 'models': []}
        except requests.exceptions.RequestException:
            return {'status': 'not_installed', 'models': []}
    
    def install_model(self, model_name: str) -> bool:
        """Install an Ollama model"""
        print(f"Starting installation of model: {model_name}")
        try:
            print(f"Running command: ollama pull {model_name}")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ Successfully installed model: {model_name}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                return True
            else:
                print(f"✗ Failed to install model: {model_name}")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Installation timeout for model: {model_name} (exceeded 300 seconds)")
            return False
        except FileNotFoundError:
            print("✗ Ollama command not found. Please make sure Ollama is installed and in your PATH.")
            return False
        except Exception as e:
            print(f"✗ Unexpected error during installation of {model_name}: {str(e)}")
            return False
    
    def generate_response(
        self, 
        user_input: str, 
        model_name: str,
        knowledge_base: Optional[Any] = None,
        chat_history: List[Dict] = None
    ) -> str:
        """Generate AI response using Ollama"""
        print(f"\n🔄 Starting AI response generation...")
        print(f"📝 User input: {user_input}")
        print(f"🤖 Model: {model_name}")
        print(f"📚 Knowledge base available: {knowledge_base is not None}")
        print(f"💬 Chat history length: {len(chat_history) if chat_history else 0}")
        
        try:
            print(f"🔗 Initializing Ollama LLM with base_url: {self.base_url}")
            llm = Ollama(model=model_name, base_url=self.base_url)
            
            # Prepare context from knowledge base
            context = ""
            if knowledge_base:
                print("🔍 Searching knowledge base for relevant documents...")
                docs = knowledge_base.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                print(f"📄 Found {len(docs)} relevant documents")
                print(f"📝 Context length: {len(context)} characters")
            else:
                print("📄 No knowledge base context available")
            
            # Build prompt
            print("🏗️ Building system prompt...")
            system_prompt = self._build_system_prompt(context)
            print("🏗️ Building conversation context...")
            conversation_context = self._build_conversation_context(chat_history)
            
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation History:\n{conversation_context}\n\n"
                f"Human: {user_input}\nAssistant:"
            )
            
            print(f"📏 Full prompt length: {len(full_prompt)} characters")
            print("🚀 Sending request to Ollama...")
            
            response = llm(full_prompt)
            
            print(f"✅ Received response from Ollama")
            print(f"📏 Response length: {len(response)} characters")
            print(f"🤖 AI Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error in generate_response: {str(e)}"
            print(error_msg)
            print(f"🔧 Error type: {type(e).__name__}")
            return (
                "I'm sorry, I encountered an error while processing your request. "
                "Please make sure Ollama is running with a model installed."
            )
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context"""
        return f"""You are a helpful AI assistant. 
        You have access to the following context from uploaded documents:
        {context}
        
        Use this context to answer questions when relevant, but also provide 
        general assistance when needed. Be conversational and helpful. 
        Keep responses concise but informative."""
    
    def _build_conversation_context(self, chat_history: List[Dict]) -> str:
        """Build conversation context from history"""
        if not chat_history:
            return ""
        
        context = ""
        for message in chat_history[-6:]:  # Last 6 messages
            role = "Human" if message["role"] == "user" else "Assistant"
            context += f"{role}: {message['content']}\n"
        
        return context 
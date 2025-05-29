"""
Enhanced Ollama Model with Source Citations and Educational Features
"""

import requests
import subprocess
import re
from typing import Dict, List, Optional, Any, Tuple
from langchain_community.llms import Ollama
import streamlit as st

class EnhancedOllamaModel:
    def __init__(self, config):
        self.config = config
        self.base_url = config.base_url
        
        # Educational prompts for different modes
        self.educational_prompts = {
            'explain': """You are an expert educator using the Feynman Technique. 
                         Explain concepts clearly and simply, using analogies and examples.""",
            'quiz': """Create interactive quiz questions with multiple choice answers 
                      and explanations for each option.""",
            'summarize': """Summarize content for students, highlighting key concepts 
                           and their relationships.""",
            'elaborate': """Provide detailed explanations with real-world applications 
                           and connections to other concepts.""",
            'practice': """Create practice problems with step-by-step solutions and hints."""
        }
    
    def check_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and get available models"""
        models = []
        status = 'not_installed'
        error_msg = None
        
        # Try API first
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                api_models = response.json().get('models', [])
                models.extend([model['name'] for model in api_models])
                status = 'running'
                print(f"✅ API detection found {len(models)} models: {models}")
        except requests.exceptions.RequestException as e:
            error_msg = f"API connection failed: {str(e)}"
            print(f"⚠️ API detection failed: {error_msg}")
        
        # Also try command line approach for more comprehensive detection
        try:
            import subprocess
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                status = 'running'
                lines = result.stdout.strip().split('\n')
                print(f"📋 Command line output:\n{result.stdout}")
                
                # Parse command line output
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if line and not line.startswith('-') and not line.startswith('NAME'):
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            base_name = model_name.split(':')[0]
                            
                            # Add both full name and base name if not already present
                            if model_name not in models:
                                models.append(model_name)
                                print(f"   Added model: {model_name}")
                            if base_name != model_name and base_name not in models:
                                models.append(base_name)
                                print(f"   Added base model: {base_name}")
                
                print(f"✅ Command line detection found {len(models)} total models: {models}")
            else:
                cmd_error = result.stderr.strip() or "Unknown command error"
                if not error_msg:
                    error_msg = f"Command failed: {cmd_error}"
                print(f"❌ Command line detection failed: {cmd_error}")
        except subprocess.TimeoutExpired:
            cmd_error = "Command timeout"
            if not error_msg:
                error_msg = cmd_error
            print(f"⏰ Command line detection timed out")
        except FileNotFoundError:
            cmd_error = "Ollama command not found"
            if not error_msg:
                error_msg = cmd_error
            print(f"❌ Ollama command not found")
        except Exception as e:
            cmd_error = f"Command error: {str(e)}"
            if not error_msg:
                error_msg = cmd_error
            print(f"❌ Command line error: {str(e)}")
        
        # Remove duplicates while preserving order
        unique_models = []
        for model in models:
            if model not in unique_models:
                unique_models.append(model)
        
        print(f"🔍 Final model detection result: status={status}, models={unique_models}")
        
        return {
            'status': status,
            'models': unique_models,
            'error': error_msg
        }
    
    def install_model(self, model_name: str) -> bool:
        """Install an Ollama model with progress feedback"""
        try:
            with st.status(f"Installing {model_name}...", expanded=True) as status:
                st.write("🔄 Starting model download...")
                
                result = subprocess.run(
                    ["ollama", "pull", model_name],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=600  # Increased timeout for larger models
                )
                
                if result.returncode == 0:
                    st.write("✅ Model installed successfully!")
                    status.update(label="Model installed!", state="complete")
                    return True
                else:
                    st.error(f"❌ Installation failed: {result.stderr}")
                    status.update(label="Installation failed", state="error")
                    return False
                    
        except subprocess.TimeoutExpired:
            st.error("⏰ Installation timeout")
            return False
        except FileNotFoundError:
            st.error("❌ Ollama not found. Please install Ollama first.")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    def generate_response(
        self, 
        user_input: str, 
        model_name: str,
        knowledge_base: Optional[Any] = None,
        chat_history: List[Dict] = None,
        mode: str = 'chat'
    ) -> str:
        """Generate AI response using Ollama with optional educational modes"""
        try:
            print(f"🎯 generate_response called:")
            print(f"   📝 Query: {user_input[:100]}...")
            print(f"   🤖 Model: {model_name}")
            print(f"   📚 Knowledge base: {'✅ Available' if knowledge_base else '❌ None'}")
            print(f"   🎨 Mode: {mode}")
            
            # First check if Ollama is running and model is available
            status = self.check_status()
            print(f"🔍 Status check result: {status}")
            
            if status['status'] != 'running':
                error_details = status.get('error', 'Unknown error')
                error_msg = f"❌ Ollama is not running: {error_details}. Please start Ollama by running 'ollama serve' in your terminal."
                print(error_msg)
                return error_msg
            
            available_models = status.get('models', [])
            if not available_models:
                error_msg = "❌ No models are installed. Please install a model first using 'ollama pull <model_name>' in your terminal."
                print(error_msg)
                return error_msg
            
            # Check if the requested model is available
            if not self.check_model_installed(model_name):
                error_msg = f"❌ Model '{model_name}' is not installed. Available models: {', '.join(available_models)}. You can install it using 'ollama pull {model_name}' in your terminal."
                print(error_msg)
                return error_msg
            
            print(f"🚀 Initializing Ollama with model: {model_name}, base_url: {self.base_url}")
            llm = Ollama(model=model_name, base_url=self.base_url)
            
            # Use educational mode if specified
            if mode in self.educational_prompts:
                print(f"🎓 Using educational mode: {mode}")
                return self.generate_educational_response(
                    user_input, llm, knowledge_base, chat_history, mode
                )
            
            # Get context with sources if knowledge base is available
            context, sources = self._get_context_with_sources(user_input, knowledge_base)
            
            if context:
                print(f"📚 Retrieved context from knowledge base:")
                print(f"   📄 Context length: {len(context)} characters")
                print(f"   🔗 Sources: {len(sources)}")
                print(f"   📝 Context preview: {context[:300]}...")
                
                # Build strong prompt that prioritizes knowledge base content
                system_prompt = f"""You are a helpful AI assistant. You have been provided with specific context from documents.

CRITICAL INSTRUCTIONS:
1. You MUST answer based on the provided context below
2. If the context contains information about the topic, use ONLY that information
3. Do NOT use your general training knowledge if it contradicts the context
4. If the context doesn't contain relevant information, clearly state "Based on the provided documents, I don't have information about this topic"
5. Always cite your sources when using information from the context

Context from uploaded documents:
{context}

Remember: The context above contains the most relevant and up-to-date information. Always prioritize it over your general knowledge."""
                
            else:
                print("📭 No context retrieved from knowledge base")
                system_prompt = "You are a helpful AI assistant."
            
            conversation_context = self._build_conversation_context(chat_history)
            
            if conversation_context:
                print(f"💬 Using conversation history: {len(conversation_context)} characters")
            
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation History:\n{conversation_context}\n\n"
                f"Human: {user_input}\nAssistant:"
            )
            
            print(f"📤 Sending request to Ollama model: {model_name}")
            print(f"🔍 Full prompt length: {len(full_prompt)} characters")
            print(f"📝 Prompt preview:\n{full_prompt[:500]}...")
            
            response = llm(full_prompt)
            print(f"📥 Received response from Ollama (length: {len(response)} chars)")
            print(f"💬 Response preview: {response[:200]}...")
            
            # Add source citations if sources were used
            if sources and context:
                print(f"🔗 Adding source citations for {len(sources)} sources")
                response = self._add_source_citations(response, sources)
            
            return response
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"❌ Cannot connect to Ollama server at {self.base_url}. Please make sure Ollama is running by executing 'ollama serve' in your terminal. Connection error: {str(e)}"
            print(error_msg)
            return error_msg
        except requests.exceptions.Timeout as e:
            error_msg = f"❌ Request to Ollama timed out. The model might be loading or the server is overloaded. Please try again in a moment. Timeout error: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            # Get more detailed error information
            error_type = type(e).__name__
            error_details = str(e)
            
            # Check if it's a model-specific error
            if 'model' in error_details.lower() and 'not found' in error_details.lower():
                error_msg = f"❌ Model '{model_name}' not found on server. Please install it with 'ollama pull {model_name}' in your terminal."
                print(error_msg)
                return error_msg
            
            print(f"❌ Detailed error in generate_response: {error_type}: {error_details}")
            st.error(f"Error generating response ({error_type}): {error_details}")
            error_msg = (
                f"❌ Error generating response: {error_type} - {error_details}\n\n"
                "Troubleshooting steps:\n"
                "1. Make sure Ollama is running: 'ollama serve'\n"
                f"2. Verify model is installed: 'ollama list'\n"
                f"3. If needed, install the model: 'ollama pull {model_name}'\n"
                "4. Check if Ollama is accessible on the correct port (default: 11434)"
            )
            print(error_msg)
            return error_msg
    
    def generate_with_sources(self, query: str, context_docs: List, model_name: str) -> Dict:
        """Generate response with source citations"""
        try:
            print(f"🎯 generate_with_sources called:")
            print(f"   📝 Query: {query[:100]}...")
            print(f"   🤖 Model: {model_name}")
            print(f"   📚 Context docs: {len(context_docs)}")
            
            # First check if Ollama is running and model is available
            status = self.check_status()
            if status['status'] != 'running':
                error_msg = f"❌ Ollama not running: {status.get('error', 'Unknown error')}"
                print(error_msg)
                return {
                    'response': f"{error_msg}. Please start Ollama by running 'ollama serve' in your terminal.",
                    'sources': [],
                    'citations': []
                }
            
            if not self.check_model_installed(model_name):
                available_models = status.get('models', [])
                error_msg = f"❌ Model '{model_name}' not installed. Available: {available_models}"
                print(error_msg)
                return {
                    'response': f"{error_msg}. Install with 'ollama pull {model_name}'.",
                    'sources': [],
                    'citations': []
                }
            
            print(f"✅ Ollama ready with model: {model_name}")
            llm = Ollama(model=model_name, base_url=self.base_url)
            
            # Format context with source tracking
            formatted_context = self._format_context_with_sources(context_docs)
            print(f"📄 Formatted context length: {len(formatted_context)} characters")
            print(f"📝 Context preview: {formatted_context[:200]}...")
            
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
IMPORTANT: You MUST base your answer on the provided context. Do NOT use your general knowledge if it contradicts the context.
Always cite your sources using [Source N] format when using information from the context.

Context:
{formatted_context}

Question: {query}

Instructions:
1. Answer ONLY based on the context provided above
2. If the context contains information about the topic, use ONLY that information
3. Cite sources using [Source N] format for every fact you mention
4. If the context doesn't contain relevant information, clearly state "Based on the provided documents, I don't have information about..."
5. Do NOT supplement with your general knowledge - stick to the context only

Answer:"""
            
            print(f"📤 Sending prompt to model (length: {len(prompt)} chars)")
            print(f"🔍 Prompt preview:\n{prompt[:500]}...")
            
            response = llm(prompt)
            print(f"📥 Received response (length: {len(response)} chars)")
            print(f"💬 Response preview: {response[:200]}...")
            
            # Extract citations and create source mapping
            sources_used = self._extract_citations(response)
            source_details = []
            
            print(f"🔗 Citations found: {sources_used}")
            
            for source_num in sources_used:
                if source_num <= len(context_docs):
                    doc = context_docs[source_num - 1]  # Convert to 0-based index
                    
                    if isinstance(doc, tuple):
                        text, metadata, score = doc[0], doc[1], doc[2] if len(doc) > 2 else 0.0
                    else:
                        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        score = 0.0
                    
                    source_details.append({
                        'index': source_num,
                        'text': text,
                        'metadata': metadata,
                        'score': score
                    })
                    print(f"   📄 Added source {source_num}: {metadata.get('filename', 'Unknown')}")
            
            result = {
                'response': response,
                'sources': source_details,
                'citations': sources_used
            }
            
            print(f"✅ Response generated with {len(source_details)} sources cited")
            return result
            
        except Exception as e:
            error_msg = f"Error generating response with sources: {str(e)}"
            print(f"❌ {error_msg}")
            st.error(error_msg)
            return {
                'response': f"❌ {error_msg}",
                'sources': [],
                'citations': []
            }
    
    def generate_educational_response(
        self, 
        query: str, 
        llm: Ollama, 
        knowledge_base: Optional[Any] = None,
        chat_history: List[Dict] = None,
        mode: str = 'explain'
    ) -> str:
        """Generate educational response with specific pedagogical approach"""
        
        context, sources = self._get_context_with_sources(query, knowledge_base)
        
        # Get educational prompt for the mode
        educational_prompt = self.educational_prompts.get(mode, self.educational_prompts['explain'])
        
        if mode == 'explain':
            prompt = f"""{educational_prompt}

Context: {context}

Topic to explain: {query}

Please follow this structure:
1. **Simple Overview**: Start with a simple, one-sentence explanation
2. **Key Concepts**: Break down the main ideas into digestible parts
3. **Analogy**: Provide a relatable comparison or metaphor
4. **Example**: Give a concrete, practical example
5. **Check Understanding**: Ask a thoughtful question to verify comprehension

Make it engaging and progressively build understanding."""

        elif mode == 'quiz':
            prompt = f"""{educational_prompt}

Context: {context}
Topic: {query}

Create an interactive quiz question following this format:

**Question**: [Clear, specific question about the topic]

**Options**:
A) [Option 1]
B) [Option 2] 
C) [Option 3]
D) [Option 4]

**Hint**: [Helpful hint without giving away the answer]

**Correct Answer**: [Letter and explanation]
**Why other options are incorrect**: [Brief explanations]"""

        elif mode == 'practice':
            prompt = f"""{educational_prompt}

Context: {context}
Skill/Topic: {query}

Create a practice exercise with:
1. **Problem Statement**: Clear, realistic scenario
2. **What to do**: Step-by-step instructions
3. **Hints**: Progressive hints (hide initially)
4. **Solution**: Complete solution with explanation
5. **Extension**: Additional practice suggestions

Make it challenging but achievable."""

        else:
            # Default to summary mode
            prompt = f"""{educational_prompt}

Context: {context}
Content to summarize: {query}

Create a student-friendly summary with:
- **Main Ideas**: 3-5 key points
- **Important Terms**: Definitions of key vocabulary
- **Connections**: How this relates to other concepts
- **Why It Matters**: Real-world relevance

Keep it concise but comprehensive."""
        
        try:
            response = llm(prompt)
            
            # Add source citations if available
            if sources and context:
                response = self._add_source_citations(response, sources)
            
            return response
            
        except Exception as e:
            return f"Error generating educational response: {str(e)}"
    
    def _get_context_with_sources(self, query: str, knowledge_base) -> Tuple[str, List]:
        """Get context and source information from knowledge base"""
        print(f"🔍 _get_context_with_sources called:")
        print(f"   📝 Query: {query[:100]}...")
        print(f"   📚 Knowledge base: {'✅ Available' if knowledge_base else '❌ None'}")
        
        if not knowledge_base:
            print("   ❌ No knowledge base provided")
            return "", []
        
        try:
            print(f"   🔧 Knowledge base type: {type(knowledge_base).__name__}")
            print(f"   📋 Available methods: {[method for method in dir(knowledge_base) if not method.startswith('_')]}")
            
            results = []
            
            # Try hybrid search first, but with error handling
            if hasattr(knowledge_base, 'hybrid_search'):
                print("   🎯 Attempting hybrid_search method...")
                try:
                    results = knowledge_base.hybrid_search(query, k=5)
                    print(f"   ✅ Hybrid search successful: {len(results)} results")
                except Exception as e:
                    print(f"   ❌ Hybrid search failed: {str(e)}")
                    print("   🔄 Falling back to similarity search...")
                    results = []
            
            # Fallback to similarity search if hybrid failed or not available
            if not results and hasattr(knowledge_base, 'similarity_search'):
                print("   🎯 Using similarity_search method as fallback")
                try:
                    results = knowledge_base.similarity_search(query, k=5)
                    print(f"   ✅ Similarity search returned {len(results)} results")
                except Exception as e:
                    print(f"   ❌ Similarity search also failed: {str(e)}")
                    results = []
            
            # Final fallback - check if we have documents directly
            if not results and hasattr(knowledge_base, 'documents'):
                print("   🎯 Attempting direct document search as last resort...")
                try:
                    documents = knowledge_base.documents
                    if documents:
                        # Simple text matching for last resort
                        query_lower = query.lower()
                        matching_docs = []
                        
                        for i, doc in enumerate(documents):
                            doc_text = str(doc).lower()
                            if any(word in doc_text for word in query_lower.split()):
                                matching_docs.append((doc, {'index': i}, 0.5))
                        
                        results = matching_docs[:5]  # Limit to 5 results
                        print(f"   ✅ Direct search found {len(results)} matching documents")
                    else:
                        print("   ❌ No documents found in knowledge base")
                except Exception as e:
                    print(f"   ❌ Direct document search failed: {str(e)}")
            
            if not results:
                print("   ⚠️ All search methods failed - no results returned")
                return "", []
            
            context_parts = []
            sources = []
            
            print(f"   📄 Processing {len(results)} search results:")
            
            for i, result in enumerate(results):
                print(f"      📄 Processing result {i+1}:")
                print(f"         📋 Type: {type(result)}")
                
                if isinstance(result, tuple) and len(result) >= 2:
                    # Enhanced knowledge base format: (text, metadata, score)
                    text = result[0]
                    metadata = result[1] if len(result) > 1 else {}
                    score = result[2] if len(result) > 2 else 0.0
                    print(f"         📝 Tuple format - Text: {len(str(text))} chars, Score: {score:.3f}")
                    
                elif hasattr(result, 'page_content'):
                    # LangChain document format
                    text = result.page_content
                    metadata = result.metadata if hasattr(result, 'metadata') else {}
                    score = 0.0
                    print(f"         📝 LangChain format - Text: {len(text)} chars")
                    
                else:
                    # String or other format
                    text = str(result)
                    metadata = {}
                    score = 0.0
                    print(f"         📝 String format - Text: {len(text)} chars")
                
                # Ensure text is string
                text = str(text)
                
                if not text or len(text.strip()) < 10:
                    print(f"         ⚠️ Skipping empty/short result")
                    continue
                
                # Add source prefix and store
                source_text = f"[Source {i+1}] {text}"
                context_parts.append(source_text)
                
                source_info = {
                    'index': i+1,
                    'text': text,
                    'metadata': metadata,
                    'score': score
                }
                sources.append(source_info)
                
                filename = metadata.get('filename', f'Document {i+1}') if metadata else f'Document {i+1}'
                print(f"         ✅ Added source {i+1}: {filename} ({len(text)} chars)")
                print(f"         📝 Preview: {text[:100]}...")
            
            context = "\n\n".join(context_parts)
            
            print(f"   🎯 Final context:")
            print(f"      📄 Total context length: {len(context)} characters")
            print(f"      🔗 Total sources: {len(sources)}")
            print(f"      📝 Context preview: {context[:300]}...")
            
            return context, sources
            
        except Exception as e:
            error_msg = f"Error retrieving context: {str(e)}"
            print(f"   ❌ {error_msg}")
            st.error(error_msg)
            return "", []
    
    def _format_context_with_sources(self, docs: List) -> str:
        """Format documents with source numbers"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            if isinstance(doc, tuple):
                text, metadata = doc[0], doc[1]
                filename = metadata.get('filename', f'Document {i}')
            elif hasattr(doc, 'page_content'):
                text = doc.page_content
                filename = doc.metadata.get('filename', f'Document {i}')
            else:
                text = str(doc)
                filename = f'Document {i}'
            
            formatted.append(f"[Source {i}] {filename}: {text}")
        
        return "\n\n".join(formatted)
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract source citations from response"""
        citations = re.findall(r'\[Source (\d+)\]', text)
        return list(set(int(c) for c in citations))
    
    def _add_source_citations(self, response: str, sources: List) -> str:
        """Add source information at the end of response"""
        if not sources:
            return response
        
        citation_text = "\n\n**Sources:**\n"
        for source in sources:
            filename = source['metadata'].get('filename', 'Unknown Document')
            score = source.get('score', 0.0)
            citation_text += f"- [Source {source['index']}] {filename} (relevance: {score:.2f})\n"
        
        return response + citation_text
    
    def _build_system_prompt(self, context: str, mode: str = 'chat') -> str:
        """Build system prompt based on mode and context"""
        base_prompt = "You are a helpful AI assistant."
        
        if mode == 'educational':
            base_prompt = "You are an expert educator. Explain concepts clearly and pedagogically."
        elif mode == 'research':
            base_prompt = "You are a research assistant. Provide detailed, well-sourced information."
        elif mode == 'casual':
            base_prompt = "You are a friendly conversational AI. Keep responses casual and engaging."
        
        if context:
            base_prompt += f"\n\nYou have access to the following context:\n{context}"
            base_prompt += ("\n\nUse this context to answer questions when relevant. "
                          "Always cite sources when using specific information from the context. "
                          "If the context doesn't contain relevant information, say so and provide "
                          "general assistance based on your training.")
        
        return base_prompt
    
    def _build_conversation_context(self, chat_history: List[Dict]) -> str:
        """Build conversation context from history"""
        if not chat_history:
            return ""
        
        context = ""
        # Use last 8 messages for more context
        for message in chat_history[-8:]:
            role = "Human" if message["role"] == "user" else "Assistant"
            content = message['content'][:500]  # Limit length
            context += f"{role}: {content}\n"
        
        return context
    
    def check_model_installed(self, model_name: str) -> bool:
        """Check if a specific model is installed"""
        if not model_name:
            return False
            
        print(f"🔍 Checking if model '{model_name}' is installed...")
        status = self.check_status()
        available_models = status.get('models', [])
        
        print(f"📋 Available models from check_status: {available_models}")
        
        # Direct match
        if model_name in available_models:
            print(f"✅ Direct match found for '{model_name}'")
            return True
        
        # Check base name (without tags)
        base_name = model_name.split(':')[0]
        if base_name in available_models:
            print(f"✅ Base name match found for '{base_name}'")
            return True
        
        # Check with common default tags
        common_tags = ['latest', '7b', '3b', '1b', '13b']
        for tag in common_tags:
            full_name = f"{base_name}:{tag}"
            if full_name in available_models:
                print(f"✅ Tag match found for '{full_name}'")
                return True
            
        # Check if any model starts with the base name
        for available_model in available_models:
            if available_model.startswith(base_name):
                print(f"✅ Prefix match found: '{available_model}' starts with '{base_name}'")
                return True
        
        print(f"❌ Model '{model_name}' not found in available models")
        return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model"""
        try:
            response = requests.get(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except:
            return {}
    
    def list_running_models(self) -> List[str]:
        """List currently running models"""
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                return []
        except:
            return [] 

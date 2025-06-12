#!/usr/bin/env python3
"""
Ollama Test and Setup Script for Enhanced RAG System

This script helps verify that Ollama is properly configured and 
can download required models if needed.
"""

import requests
import json
import subprocess
import sys
import time
from typing import List, Dict

class OllamaSetupHelper:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [model["name"] for model in response.json()["models"]]
            return models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def test_embedding_model(self, model: str = "nomic-embed-text") -> bool:
        """Test if embedding model works"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": "test text for embedding"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                return embedding is not None and len(embedding) > 0
            else:
                print(f"Embedding test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error testing embedding model: {e}")
            return False
    
    def test_chat_model(self, model: str = "llama3.1") -> bool:
        """Test if chat model works"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "Say hello in exactly 3 words.",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get("response")
                return result is not None and len(result.strip()) > 0
            else:
                print(f"Chat test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error testing chat model: {e}")
            return False
    
    def pull_model(self, model: str) -> bool:
        """Pull a model using Ollama CLI"""
        try:
            print(f"Downloading {model}... (this may take a while)")
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded {model}")
                return True
            else:
                print(f"❌ Failed to download {model}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout downloading {model}")
            return False
        except FileNotFoundError:
            print("❌ Ollama CLI not found. Please install Ollama first.")
            return False
        except Exception as e:
            print(f"❌ Error downloading {model}: {e}")
            return False

def main():
    print("🚀 Ollama Setup Helper for Enhanced RAG System")
    print("=" * 50)
    
    helper = OllamaSetupHelper()
    
    # Step 1: Check if Ollama is running
    print("\n1. Checking Ollama server...")
    if helper.check_ollama_running():
        print("✅ Ollama server is running")
    else:
        print("❌ Ollama server is not running")
        print("💡 Please start Ollama with: ollama serve")
        print("💡 Or visit https://ollama.ai for installation instructions")
        return False
    
    # Step 2: Get installed models
    print("\n2. Checking installed models...")
    installed_models = helper.get_installed_models()
    
    if installed_models:
        print("✅ Found installed models:")
        for model in installed_models:
            print(f"   - {model}")
    else:
        print("ℹ️  No models found")
    
    # Step 3: Check required models
    print("\n3. Checking required models...")
    
    required_models = {
        "embedding": ["nomic-embed-text", "mxbai-embed-large"],
        "chat": ["llama3.1", "llama3.2", "mistral", "llama2"]
    }
    
    # Check embedding models
    embedding_model_found = None
    for model in required_models["embedding"]:
        if any(model in installed for installed in installed_models):
            embedding_model_found = model
            break
    
    if embedding_model_found:
        print(f"✅ Embedding model found: {embedding_model_found}")
    else:
        print("❌ No embedding model found")
        print("💡 Recommended: nomic-embed-text")
        
        if input("Download nomic-embed-text? (y/n): ").lower().startswith('y'):
            if helper.pull_model("nomic-embed-text"):
                embedding_model_found = "nomic-embed-text"
                installed_models.append("nomic-embed-text")
    
    # Check chat models
    chat_model_found = None
    for model in required_models["chat"]:
        if any(model in installed for installed in installed_models):
            chat_model_found = model
            break
    
    if chat_model_found:
        print(f"✅ Chat model found: {chat_model_found}")
    else:
        print("❌ No chat model found")
        print("💡 Recommended: llama3.1")
        
        if input("Download llama3.1? (y/n): ").lower().startswith('y'):
            if helper.pull_model("llama3.1"):
                chat_model_found = "llama3.1"
                installed_models.append("llama3.1")
    
    # Step 4: Test the models
    print("\n4. Testing models...")
    
    if embedding_model_found:
        print(f"Testing embedding model: {embedding_model_found}")
        if helper.test_embedding_model(embedding_model_found):
            print("✅ Embedding model works correctly")
        else:
            print("❌ Embedding model test failed")
    
    if chat_model_found:
        print(f"Testing chat model: {chat_model_found}")
        if helper.test_chat_model(chat_model_found):
            print("✅ Chat model works correctly")
        else:
            print("❌ Chat model test failed")
    
    # Step 5: Final recommendations
    print("\n5. Setup Summary")
    print("=" * 30)
    
    if embedding_model_found and chat_model_found:
        print("🎉 Great! Your Ollama setup is ready for the Enhanced RAG System")
        print(f"\nRecommended configuration:")
        print(f"   - Embedding Model: {embedding_model_found}")
        print(f"   - Chat Model: {chat_model_found}")
        print(f"   - Ollama URL: {helper.base_url}")
        
        print(f"\n🚀 You can now run: streamlit run enhanced_rag_app.py")
        
    else:
        print("⚠️  Setup incomplete. Please install required models:")
        if not embedding_model_found:
            print("   - ollama pull nomic-embed-text")
        if not chat_model_found:
            print("   - ollama pull llama3.1")
    
    # Additional model suggestions
    print("\n6. Additional Model Suggestions")
    print("=" * 35)
    print("For better performance, consider these models:")
    print("\nEmbedding models:")
    print("   - nomic-embed-text (recommended, fast)")
    print("   - mxbai-embed-large (higher quality, slower)")
    
    print("\nChat models:")
    print("   - llama3.1 (recommended, good balance)")
    print("   - llama3.2 (faster, lighter)")
    print("   - mistral (good alternative)")
    print("   - codellama (better for technical documents)")
    
    print("\nTo download: ollama pull <model-name>")
    
    return True

def interactive_test():
    """Interactive testing mode"""
    print("\n🧪 Interactive Test Mode")
    print("=" * 25)
    
    helper = OllamaSetupHelper()
    
    if not helper.check_ollama_running():
        print("❌ Ollama server is not running")
        return
    
    models = helper.get_installed_models()
    if not models:
        print("❌ No models installed")
        return
    
    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    while True:
        model_choice = input("\nSelect model to test (number or name, 'q' to quit): ").strip()
        
        if model_choice.lower() == 'q':
            break
        
        try:
            if model_choice.isdigit():
                model = models[int(model_choice) - 1]
            else:
                model = model_choice
        except (IndexError, ValueError):
            print("Invalid selection")
            continue
        
        print(f"\nTesting {model}...")
        
        # Test as embedding model
        print("Testing as embedding model...")
        if helper.test_embedding_model(model):
            print("✅ Works as embedding model")
        else:
            print("❌ Does not work as embedding model")
        
        # Test as chat model
        print("Testing as chat model...")
        if helper.test_chat_model(model):
            print("✅ Works as chat model")
        else:
            print("❌ Does not work as chat model")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        main()

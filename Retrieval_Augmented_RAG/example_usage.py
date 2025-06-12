# Example usage script for the Enhanced RAG System
# This shows how to use the system programmatically without Streamlit

import sys
import os
from enhanced_rag_app import EnhancedRAGSystem

def main():
    """
    Example usage of the Enhanced RAG System
    """
    
    # Initialize the RAG system
    print("Initializing Enhanced RAG System...")
    rag_system = EnhancedRAGSystem(ollama_base_url="http://localhost:11434")
    
    # Path to your PDF file
    pdf_path = "example_document.pdf"  # Replace with your PDF path
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please provide a valid PDF file path.")
        return
    
    # Configuration parameters
    config = {
        'chunk_size': 800,
        'embedding_model': 'nomic-embed-text',
        'chat_model': 'llama3.1',
        'irrelevant_penalty': 0.2,
        'max_segment_length': 20,
        'total_max_length': 30,
        'min_segment_value': 0.2,
        'top_k': 10
    }
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Configuration: {config}")
    
    try:
        # Process the document
        print("\n1. Processing document...")
        with open(pdf_path, 'rb') as file:
            chunks, vector_store, doc_info = rag_system.process_document(
                file, 
                config['chunk_size'], 
                config['embedding_model']
            )
        
        if chunks is None:
            print("Error: Failed to process document")
            return
        
        print(f"   ✓ Extracted {doc_info['total_text_length']:,} characters")
        print(f"   ✓ Created {doc_info['chunk_count']} chunks")
        
        # Example queries
        queries = [
            "What is the main topic of this document?",
            "Can you summarize the key findings?",
            "What are the conclusions or recommendations?",
            # Add your own queries here
        ]
        
        print("\n2. Running example queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print('='*60)
            
            # Method 1: RSE (Relevant Segment Extraction)
            print("\n🎯 RSE Method:")
            print("-" * 40)
            
            rse_result = rag_system.rag_with_rse(
                chunks=chunks,
                vector_store=vector_store,
                query=query,
                embedding_model=config['embedding_model'],
                chat_model=config['chat_model'],
                irrelevant_chunk_penalty=config['irrelevant_penalty'],
                max_segment_length=config['max_segment_length'],
                total_max_length=config['total_max_length'],
                min_segment_value=config['min_segment_value']
            )
            
            print(f"Found {len(rse_result['segments'])} segments")
            print(f"Response: {rse_result['response']}")
            
            # Method 2: Standard Top-K Retrieval
            print("\n📝 Standard Top-K Method:")
            print("-" * 40)
            
            standard_result = rag_system.standard_top_k_retrieval(
                chunks=chunks,
                vector_store=vector_store,
                query=query,
                embedding_model=config['embedding_model'],
                chat_model=config['chat_model'],
                k=config['top_k']
            )
            
            print(f"Retrieved {len(standard_result['chunks'])} chunks")
            print(f"Response: {standard_result['response']}")
            
            # Show segment details for RSE
            print(f"\n📋 RSE Segment Details:")
            for j, segment in enumerate(rse_result['segments']):
                start, end = segment['segment_range']
                score = rse_result['segment_scores'][j]
                print(f"  Segment {j+1}: Chunks {start}-{end-1} (Score: {score:.4f})")
                print(f"  Preview: {segment['text'][:100]}...")
        
        print(f"\n{'='*60}")
        print("✅ Processing complete!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

def interactive_mode():
    """
    Interactive mode for asking custom questions
    """
    print("\n🤖 Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    
    # Initialize system
    rag_system = EnhancedRAGSystem()
    
    # Get PDF path
    pdf_path = input("\nEnter PDF file path: ").strip()
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return
    
    # Process document
    print("Processing document...")
    try:
        with open(pdf_path, 'rb') as file:
            chunks, vector_store, doc_info = rag_system.process_document(
                file, chunk_size=800, embedding_model='nomic-embed-text'
            )
        
        if chunks is None:
            print("Failed to process document")
            return
        
        print(f"✓ Document processed: {doc_info['chunk_count']} chunks")
        
    except Exception as e:
        print(f"Error processing document: {e}")
        return
    
    # Interactive query loop
    while True:
        print("\n" + "-"*50)
        query = input("Enter your question (or 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        elif query.lower() == 'help':
            print("\nCommands:")
            print("  quit/exit/q - Exit interactive mode")
            print("  help - Show this help")
            print("  Just type a question to get an answer")
            continue
        elif not query:
            continue
        
        try:
            # Use RSE method by default in interactive mode
            result = rag_system.rag_with_rse(
                chunks=chunks,
                vector_store=vector_store,
                query=query,
                embedding_model='nomic-embed-text',
                chat_model='llama3.1'
            )
            
            print(f"\n🎯 Answer (using {len(result['segments'])} segments):")
            print(result['response'])
            
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    print("Enhanced RAG System - Example Usage")
    print("==================================")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            print("Usage:")
            print("  python example_usage.py                # Run examples")
            print("  python example_usage.py --interactive  # Interactive mode")
    else:
        main()

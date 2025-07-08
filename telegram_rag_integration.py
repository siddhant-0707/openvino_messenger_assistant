from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ov_langchain_helper import OpenVINOBgeEmbeddings
import openvino as ov

class TelegramRAGIntegration:
    """Integration class for Telegram RAG system"""
    
    def __init__(
        self, 
        embedding_model=None,
        embedding_model_name=None,
        vector_store_path="telegram_vector_store",
        chunk_size=500,
        chunk_overlap=50
    ):
        """
        Initialize the RAG integration
        
        Args:
            embedding_model: Pre-initialized embedding model instance
            embedding_model_name: Path to embedding model (alternative to embedding_model)
            vector_store_path: Path to store vector embeddings
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        if embedding_model is not None:
            self.embedding = embedding_model
        elif embedding_model_name is not None:
            self.embedding = OpenVINOBgeEmbeddings(
                model_path=str(embedding_model_name),
                encode_kwargs={
                    "normalize_embeddings": True,
                }
            )
        else:
            raise ValueError("Either embedding_model or embedding_model_name must be provided")
            
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Try to load existing vector store
        try:
            self.vectorstore = FAISS.load_local(
                vector_store_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded existing vector store from {vector_store_path}")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vectorstore = None
    
    def process_telegram_data_dir(self, data_dir="telegram_data"):
        """Process all telegram data files in the directory"""
        # Get all JSON files in the data directory
        data_path = Path(data_dir)
        json_files = list(data_path.glob("telegram_messages_*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No telegram message files found in {data_dir}")
            
        # Process each file
        all_documents = []
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                messages = json.load(f)
            
            documents = self._create_documents_from_messages(messages)
            all_documents.extend(documents)
            
        # Create or update vector store
        if not all_documents:
            print("No documents to process")
            return
            
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(all_documents, self.embedding)
            self.vectorstore.save_local(self.vector_store_path)
        else:
            self.vectorstore.add_documents(all_documents)
            self.vectorstore.save_local(self.vector_store_path)
            
        print(f"Processed {len(all_documents)} documents into vector store")
    
    def _create_documents_from_messages(self, messages):
        """Convert messages to documents for the vector store"""
        documents = []
        
        for msg in messages:
            # Basic message metadata
            metadata = {
                "channel": msg["channel"],
                "date": msg["date"],
                "message_id": msg.get("id", ""),
                "views": msg.get("views", 0),
                "forwards": msg.get("forwards", 0),
                "type": "message"
            }
            
            # Process main message
            if msg["text"]:
                chunks = self.text_splitter.split_text(msg["text"])
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
            
            # Process article if present
            if "article" in msg and msg["article"].get("text"):
                article = msg["article"]
                article_metadata = metadata.copy()
                article_metadata["type"] = "article"
                article_metadata["article_title"] = article.get("title", "")
                article_metadata["article_url"] = article.get("url", "")
                
                # Skip articles that couldn't be extracted properly
                if article.get("extracted") is False:
                    # Still create a document with the article URL and title
                    article_doc_metadata = article_metadata.copy()
                    article_doc_metadata["chunk_id"] = 0
                    article_doc_metadata["extraction_failed"] = True
                    
                    # Use the article text (which contains our placeholder message)
                    documents.append(Document(page_content=article["text"], metadata=article_doc_metadata))
                    continue
                
                # Process normal articles
                article_chunks = self.text_splitter.split_text(article["text"])
                for i, chunk in enumerate(article_chunks):
                    article_doc_metadata = article_metadata.copy()
                    article_doc_metadata["chunk_id"] = i
                    documents.append(Document(page_content=chunk, metadata=article_doc_metadata))
        
        return documents
    
    def query_messages(self, query, k=5, filter_dict=None):
        """Query the vector store for relevant messages"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please process messages first.")
            
        results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        return results
    
    def answer_question(
        self, 
        question, 
        llm, 
        retriever=None, 
        k=5, 
        filter_dict=None,
        show_retrieved=False,
        reranker=None
    ) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using RAG
        
        Args:
            question: The question to answer
            llm: The language model to use
            retriever: Optional pre-configured retriever
            k: Number of documents to retrieve
            filter_dict: Filter for retrieval
            show_retrieved: Whether to include retrieved documents in the result
            reranker: Optional reranker to improve retrieval quality
            
        Returns:
            Answer string or dict with answer and context
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please process messages first.")
        
        try:
            # Create retriever if not provided
            if retriever is None:
                # Use standard similarity search
                vector_search_top_k = k * 2 if reranker else k  # Retrieve more for reranking
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": vector_search_top_k, "filter": filter_dict},
                    search_type="similarity"
                )
                
                # Add reranking if available
                if reranker:
                    from langchain.retrievers import ContextualCompressionRetriever
                    reranker.top_n = k
                    retriever = ContextualCompressionRetriever(
                        base_compressor=reranker,
                        base_retriever=retriever
                    )
            
            # Try to get model-specific prompt template
            rag_prompt_template = None
            try:
                from llm_config import SUPPORTED_LLM_MODELS
                for language, models in SUPPORTED_LLM_MODELS.items():
                    for model_id, config in models.items():
                        if hasattr(llm, "model_name") and model_id in str(llm.model_name) and "rag_prompt_template" in config:
                            rag_prompt_template = config["rag_prompt_template"]
                            break
                    if rag_prompt_template:
                        break
            except (ImportError, AttributeError):
                pass
            
            # Use default prompt if model-specific one is not available
            if not rag_prompt_template:
                rag_prompt_template = """
                You are a helpful assistant that answers questions based on Telegram messages and news articles.
                
                CONTEXT:
                {context}
                
                QUESTION:
                {input}
                
                INSTRUCTIONS:
                1. Answer the question directly and concisely based on the context provided.
                2. If the context contains relevant information, use it to provide a detailed answer.
                3. If the context doesn't contain enough information to fully answer the question, provide whatever partial information is available.
                4. Focus on extracting key facts and insights from the context.
                5. Don't be overly cautious - if there's information in the context that's relevant, use it confidently.
                
                ANSWER:
                """
            
            # Create prompt
            prompt = PromptTemplate.from_template(rag_prompt_template)
            
            # Create RAG chain
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            # Run the chain
            response = rag_chain.invoke({"input": question})
            
            # Return the result
            if show_retrieved and "context" in response:
                return {
                    "answer": response["answer"],
                    "context_docs": response["context"]
                }
            else:
                return response["answer"]
                
        except Exception as e:
            import traceback
            error_msg = f"Error answering question: {str(e)}\n{traceback.format_exc()}"
            if show_retrieved:
                return {"answer": error_msg, "context_docs": []}
            else:
                return error_msg

# Example usage:
def main():
    # Initialize the integration
    rag = TelegramRAGIntegration()
    
    # Process all telegram data
    rag.process_telegram_data_dir()
    
    # Example query
    query = "What are the latest announcements?"
    results = rag.query_messages(query)
    
    # Print results
    for doc in results:
        print(f"\nSource: {doc.metadata['source']}")
        print(f"Date: {doc.metadata['date']}")
        print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 
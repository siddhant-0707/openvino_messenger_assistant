from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ov_langchain_helper import OpenVINOBgeEmbeddings
import openvino as ov

class TelegramRAGIntegration:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        vector_store_path: str = "telegram_vector_store",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG integration for Telegram messages using OpenVINO
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model to use
            vector_store_path: Path to store/load the vector store
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.vector_store_path = Path(vector_store_path)
        
        # Initialize OpenVINO embeddings
        ov_model_kwargs = {"device_name": "CPU"}
        self.embeddings = OpenVINOBgeEmbeddings(
            model_path=embedding_model_name,
            model_kwargs=ov_model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        self.vectorstore = None
        if self.vector_store_path.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=str(self.vector_store_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing vector store from {self.vector_store_path}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vectorstore = None
            
    def process_message(self, message: Dict) -> List[Document]:
        """Convert a Telegram message into Documents"""
        documents = []
        
        # Base metadata
        metadata = {
            "source": f"telegram_channel_{message['channel']}",
            "message_id": message["message_id"],
            "date": message["date"],
            "channel": message["channel"],
            "views": message.get("views", 0),
            "forwards": message.get("forwards", 0)
        }
        
        # Create document for message text
        if message["text"]:
            documents.append(Document(
                page_content=message["text"],
                metadata=metadata.copy()
            ))
        
        # Create document for article content if available
        if "article" in message and message["article"].get("text"):
            article = message["article"]
            article_metadata = metadata.copy()
            article_metadata.update({
                "article_title": article.get("title"),
                "article_url": article.get("url"),
                "article_authors": article.get("authors"),
                "article_publish_date": article.get("publish_date"),
                "article_top_image": article.get("top_image")
            })
            
            # Split article content into chunks
            article_chunks = self.text_splitter.split_text(article["text"])
            for i, chunk in enumerate(article_chunks):
                chunk_metadata = article_metadata.copy()
                chunk_metadata["chunk_index"] = i
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return documents
    
    def load_messages(self, json_file: Path) -> List[Dict]:
        """Load messages from a JSON file"""
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def add_messages_to_vectorstore(
        self,
        messages: List[Dict],
        batch_size: int = 100
    ):
        """Process messages and add them to the vector store"""
        # Convert messages to documents
        all_documents = []
        for msg in messages:
            all_documents.extend(self.process_message(msg))
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(all_documents)
        
        # Initialize vector store if needed
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                chunks[:batch_size],
                self.embeddings
            )
            chunks = chunks[batch_size:]
            
        # Add remaining chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            
        # Save the updated vector store
        self.save_vectorstore()
        
    def save_vectorstore(self):
        """Save the vector store to disk"""
        if self.vectorstore is not None:
            self.vectorstore.save_local(
                folder_path=str(self.vector_store_path)
            )
            print(f"Saved vector store to {self.vector_store_path}")
        else:
            print("No vector store to save")
        
    def process_telegram_data_dir(
        self,
        data_dir: str = "telegram_data",
        batch_size: int = 100
    ):
        """Process all JSON files in the telegram data directory"""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        for json_file in data_path.glob("telegram_messages_*.json"):
            messages = self.load_messages(json_file)
            self.add_messages_to_vectorstore(messages, batch_size)
            
    def query_messages(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Document]:
        """
        Query the vector store for relevant messages
        
        Args:
            query: The search query
            k: Number of results to return
            filter_dict: Optional filter criteria (e.g., {"channel": "specific_channel"})
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized with any messages")
            
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

    def answer_question(
        self,
        question: str,
        llm: Any,
        k: int = 5,
        filter_dict: Dict = None
    ) -> str:
        """
        Answer a question about Telegram messages using RAG
        
        Args:
            question: The question to answer
            llm: The language model to use for generation
            k: Number of relevant messages to retrieve
            filter_dict: Optional filter criteria for messages
            
        Returns:
            Generated answer based on retrieved context
        """
        # Retrieve relevant messages
        relevant_docs = self.query_messages(question, k=k, filter_dict=filter_dict)
        
        # Prepare context from retrieved messages
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt template
        prompt_template = """You are a helpful assistant that answers questions about Telegram messages.
        Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.
        Keep the answer concise and relevant to the question.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        # Format prompt
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer using invoke method
        return llm.invoke(prompt)

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
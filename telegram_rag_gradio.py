import os
import asyncio
import gradio as gr
from dotenv import load_dotenv
from telegram_ingestion import TelegramChannelIngestion
from telegram_rag_integration import TelegramRAGIntegration
from pathlib import Path
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer, AutoModel
from ov_langchain_helper import OpenVINOLLM
import json
from datetime import datetime
from article_processor import ArticleProcessor

# Load environment variables
load_dotenv()

def download_and_convert_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Download and convert the model to OpenVINO format"""
    model_path = Path(model_name.split("/")[-1])
    if not model_path.exists():
        print(f"Downloading and converting model {model_name}...")
        model_path.mkdir(exist_ok=True)
        
        # Download and convert tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        convert_tokenizer(tokenizer, model_path)
        
        # Download and convert model
        model = AutoModel.from_pretrained(model_name)
        ov_model = ov.convert_model(model)
        ov.save_model(ov_model, model_path / "openvino_model.xml")
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")
    
    return str(model_path)

# Download and convert model
model_path = download_and_convert_model()

# Initialize RAG system and article processor
rag = TelegramRAGIntegration(
    embedding_model_name=model_path,
    vector_store_path="telegram_vector_store",
    chunk_size=500,
    chunk_overlap=50
)
article_processor = ArticleProcessor()

# Initialize LLM
llm = OpenVINOLLM.from_model_path(
    model_path="qwen2.5-3b-instruct/INT4_compressed_weights",
    device="CPU"
)

def format_message(msg):
    """Format a message for display"""
    date = datetime.fromisoformat(msg["date"]).strftime("%Y-%m-%d %H:%M:%S")
    output = f"""
Channel: {msg['channel']}
Date: {date}
Views: {msg.get('views', 'N/A')}
Forwards: {msg.get('forwards', 'N/A')}
Message: {msg['text'][:200]}...
{'...' if len(msg['text']) > 200 else ''}
"""
    
    # Add article information if available
    if 'article' in msg and msg['article'].get('text'):
        article = msg['article']
        output += f"""
Article Title: {article.get('title', 'N/A')}
Article URL: {article.get('url', 'N/A')}
Article Content: {article['text'][:200]}...
{'...' if len(article['text']) > 200 else ''}
"""
    
    return output

async def download_messages(channels_str: str, limit: int, hours: int) -> tuple[str, str]:
    """Download messages from specified channels"""
    channels = [c.strip() for c in channels_str.split(",") if c.strip()]
    if not channels:
        return "Please provide at least one channel name", ""
    
    try:
        ingestion = TelegramChannelIngestion(
            api_id=os.getenv("TELEGRAM_API_ID"),
            api_hash=os.getenv("TELEGRAM_API_HASH")
        )
        
        await ingestion.start()
        try:
            messages = await ingestion.process_channels(
                channels,
                limit_per_channel=limit,
                since_hours=hours
            )
            
            # Process messages to extract article content
            processed_messages = article_processor.process_messages(messages)
            
            # Format messages for display
            formatted_messages = "\n\n".join(format_message(msg) for msg in processed_messages[:5])  # Show first 5 messages
            if len(processed_messages) > 5:
                formatted_messages += f"\n\n... and {len(processed_messages) - 5} more messages"
                
            return f"Successfully downloaded and processed {len(processed_messages)} messages from {len(channels)} channels", formatted_messages
        finally:
            await ingestion.stop()
    except Exception as e:
        return f"Error downloading messages: {str(e)}", ""

def process_messages() -> str:
    """Process downloaded messages into vector store"""
    try:
        rag.process_telegram_data_dir()
        return "Successfully processed messages into vector store"
    except Exception as e:
        return f"Error processing messages: {str(e)}"

def query_messages(query: str, channel: str, num_results: int) -> str:
    """Query the vector store for relevant messages"""
    try:
        filter_dict = {"channel": channel} if channel else None
        results = rag.query_messages(query, k=num_results, filter_dict=filter_dict)
        
        output = []
        for i, doc in enumerate(results, 1):
            output.append(f"Result {i}:")
            output.append(f"Channel: {doc.metadata['channel']}")
            output.append(f"Date: {doc.metadata['date']}")
            
            # Check if this is an article result
            if 'article' in doc.metadata:
                output.append(f"Article Title: {doc.metadata.get('article_title', 'N/A')}")
                output.append(f"Article URL: {doc.metadata.get('article_url', 'N/A')}")
            
            output.append(f"Content: {doc.page_content[:200]}...")
            output.append("")
            
        return "\n".join(output) if output else "No results found"
    except Exception as e:
        return f"Error querying messages: {str(e)}"

def answer_question(
    question: str,
    channel: str,
    temperature: float,
    num_context: int
) -> str:
    """Answer questions about Telegram messages using RAG"""
    try:
        filter_dict = {"channel": channel} if channel else None
        
        # Update LLM configuration
        llm.config.temperature = temperature
        llm.config.top_p = 0.9
        llm.config.top_k = 50
        llm.config.repetition_penalty = 1.1
        
        answer = rag.answer_question(
            question=question,
            llm=llm,
            k=num_context,
            filter_dict=filter_dict
        )
        return answer
    except Exception as e:
        return f"Error answering question: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Telegram RAG System") as demo:
    gr.Markdown("# Telegram RAG System")
    
    with gr.Tab("Download Messages"):
        gr.Markdown("## Download Messages from Telegram Channels")
        channels_input = gr.Textbox(
            label="Channel Names (comma-separated)",
            placeholder="Enter channel names without @ symbol (e.g., guardian, bloomberg)"
        )
        limit_input = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Messages per Channel"
        )
        hours_input = gr.Slider(
            minimum=1,
            maximum=168,
            value=24,
            step=1,
            label="Hours to Look Back"
        )
        download_btn = gr.Button("Download Messages")
        download_status = gr.Textbox(label="Download Status")
        download_preview = gr.Textbox(label="Message Preview", lines=10)
        
    with gr.Tab("Process Messages"):
        gr.Markdown("## Process Downloaded Messages")
        process_btn = gr.Button("Process Messages")
        process_output = gr.Textbox(label="Processing Status")
        
    with gr.Tab("Query Messages"):
        gr.Markdown("## Query Processed Messages")
        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query"
        )
        channel_filter = gr.Dropdown(
            choices=["", "guardian", "bloomberg"],  # Add more channels as needed
            label="Filter by Channel (Optional)"
        )
        num_results = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Number of Results"
        )
        query_btn = gr.Button("Search")
        query_output = gr.Textbox(label="Search Results", lines=10)
    
    with gr.Tab("Question Answering"):
        gr.Markdown("## Ask Questions About Messages")
        question_input = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the Telegram messages"
        )
        qa_channel_filter = gr.Dropdown(
            choices=["", "guardian", "bloomberg"],  # Add more channels as needed
            label="Filter by Channel (Optional)"
        )
        temperature_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature (controls creativity)"
        )
        context_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of Messages for Context"
        )
        qa_btn = gr.Button("Get Answer")
        qa_output = gr.Textbox(label="Answer", lines=10)
    
    # Set up event handlers
    download_btn.click(
        fn=lambda c, l, h: asyncio.run(download_messages(c, l, h)),
        inputs=[channels_input, limit_input, hours_input],
        outputs=[download_status, download_preview]
    )
    
    process_btn.click(
        fn=process_messages,
        inputs=[],
        outputs=process_output
    )
    
    query_btn.click(
        fn=query_messages,
        inputs=[query_input, channel_filter, num_results],
        outputs=query_output
    )
    
    qa_btn.click(
        fn=answer_question,
        inputs=[
            question_input,
            qa_channel_filter,
            temperature_slider,
            context_slider
        ],
        outputs=qa_output
    )

if __name__ == "__main__":
    demo.launch() 
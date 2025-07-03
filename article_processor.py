import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import re
from urllib.parse import urlparse
import time
from newspaper import Article
import logging

class ArticleProcessor:
    def __init__(self):
        """Initialize the article processor"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def extract_url(self, text: str) -> Optional[str]:
        """Extract URL from message text"""
        # Look for markdown-style links [text](url)
        markdown_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', text)
        if markdown_match:
            return markdown_match.group(2)
            
        # Look for plain URLs
        url_match = re.search(r'https?://\S+', text)
        if url_match:
            return url_match.group(0)
            
        return None
        
    def process_article(self, url: str) -> Dict:
        """Process an article URL and extract its content"""
        try:
            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract main content
            content = article.text
            
            # If newspaper3k fails to get content, try BeautifulSoup
            if not content:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Try to find main content
                main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|article|post'))
                if main_content:
                    content = main_content.get_text(separator='\n', strip=True)
                else:
                    content = soup.get_text(separator='\n', strip=True)
            
            # Clean up content
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            
            return {
                'url': url,
                'title': article.title,
                'text': content,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image
            }
            
        except Exception as e:
            logging.error(f"Error processing article {url}: {str(e)}")
            return {
                'url': url,
                'title': None,
                'text': None,
                'error': str(e)
            }
            
    def process_message(self, message: Dict) -> Dict:
        """Process a Telegram message and extract article content"""
        url = self.extract_url(message['text'])
        if not url:
            return message
            
        # Add article content to message
        article_data = self.process_article(url)
        message['article'] = article_data
        
        return message
        
    def process_messages(self, messages: list) -> list:
        """Process multiple messages and extract article content"""
        processed_messages = []
        for msg in messages:
            processed_msg = self.process_message(msg)
            processed_messages.append(processed_msg)
            # Add a small delay to avoid overwhelming servers
            time.sleep(1)
        return processed_messages 
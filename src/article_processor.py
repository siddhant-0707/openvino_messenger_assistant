import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
import re
from urllib.parse import urlparse
import time
from newspaper import Article, ArticleException
import logging
import random

class ArticleProcessor:
    def __init__(self):
        """Initialize the article processor"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        # Sites that commonly block scraping
        self.blocked_domains = [
            'bloomberg.com', 
            'ft.com', 
            'wsj.com', 
            'nytimes.com',
            'economist.com',
            'washingtonpost.com',
            'medium.com',
            'forbes.com'
        ]
        
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
    
    def is_blocked_domain(self, url: str) -> bool:
        """Check if the URL belongs to a domain known to block scraping"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        return any(blocked in domain for blocked in self.blocked_domains)
    
    def extract_title_from_text(self, text: str) -> str:
        """Extract a potential title from the message text"""
        # Look for markdown-style links [text](url)
        markdown_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', text)
        if markdown_match:
            return markdown_match.group(1)
        
        # Look for bold text as potential title
        bold_match = re.search(r'\*\*([^*]+)\*\*', text)
        if bold_match:
            return bold_match.group(1)
        
        # Try to find a title in the first line
        lines = text.strip().split('\n')
        if lines and len(lines[0]) < 200:  # Reasonable title length
            return lines[0]
            
        return "Untitled Article"
        
    def process_article(self, url: str, message_text: str = "") -> Dict:
        """Process an article URL and extract its content"""
        try:
            # For domains known to block scraping, use a simplified approach
            if self.is_blocked_domain(url):
                return self.process_blocked_article(url, message_text)
                
            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract main content
            content = article.text
            
            # If newspaper3k fails to get content, try BeautifulSoup
            if not content:
                content = self.extract_with_beautifulsoup(url)
            
            # Clean up content
            if content:
                content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            
            return {
                'url': url,
                'title': article.title or self.extract_title_from_text(message_text),
                'text': content or "Content extraction failed. This site may block automated access.",
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'extracted': bool(content)
            }
            
        except Exception as e:
            logging.warning(f"Error processing article {url}: {str(e)}")
            return self.process_blocked_article(url, message_text)
    
    def extract_with_beautifulsoup(self, url: str) -> str:
        """Extract content using BeautifulSoup as a fallback"""
        try:
            # Add a random delay to avoid detection
            time.sleep(random.uniform(0.5, 2.0))
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Try to find main content
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', class_=re.compile(r'content|article|post|story'))
            )
            
            if main_content:
                # Extract paragraphs for cleaner text
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                else:
                    content = main_content.get_text(separator='\n', strip=True)
            else:
                content = soup.get_text(separator='\n', strip=True)
                
            return content
        except Exception as e:
            logging.warning(f"BeautifulSoup extraction failed for {url}: {str(e)}")
            return ""
    
    def process_blocked_article(self, url: str, message_text: str) -> Dict:
        """Process an article from a site that blocks scraping"""
        # Extract title from the message text if possible
        title = self.extract_title_from_text(message_text)
        
        # Get domain for reference
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Try to extract some content from the message text
        message_content = ""
        if message_text:
            # Remove the title and URL from the message
            cleaned_text = message_text
            if title in cleaned_text:
                cleaned_text = cleaned_text.replace(title, "")
            if url in cleaned_text:
                cleaned_text = cleaned_text.replace(url, "")
                
            # Clean up and use the message text as a summary
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            if cleaned_text:
                message_content = f"Message summary: {cleaned_text}"
        
        # Create a placeholder with the information we have
        return {
            'url': url,
            'title': title,
            'text': message_content or f"This article from {domain} could not be fully extracted as the site restricts automated access.",
            'authors': [],
            'publish_date': None,
            'top_image': None,
            'extracted': False
        }
            
    def process_message(self, message: Dict) -> Dict:
        """Process a Telegram message and extract article content"""
        url = self.extract_url(message['text'])
        if not url:
            return message
            
        # Add article content to message
        article_data = self.process_article(url, message['text'])
        message['article'] = article_data
        
        return message
        
    def process_messages(self, messages: List[Dict]) -> List[Dict]:
        """Process multiple messages and extract article content"""
        processed_messages = []
        for msg in messages:
            try:
                processed_msg = self.process_message(msg)
                processed_messages.append(processed_msg)
                # Add a small delay to avoid overwhelming servers
                time.sleep(random.uniform(0.2, 0.8))
            except Exception as e:
                logging.error(f"Failed to process message: {str(e)}")
                # Still add the original message even if processing failed
                processed_messages.append(msg)
                
        return processed_messages 
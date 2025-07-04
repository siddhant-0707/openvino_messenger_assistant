from telethon import TelegramClient, events
from telethon.tl.types import Channel
from typing import List, Dict
import os
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path

class TelegramChannelIngestion:
    def __init__(
        self, 
        api_id: str, 
        api_hash: str,
        session_name: str = "telegram_session",
        storage_dir: str = "telegram_data"
    ):
        """
        Initialize Telegram client for channel message ingestion
        
        Args:
            api_id: Telegram API ID from https://my.telegram.org/apps
            api_hash: Telegram API hash from https://my.telegram.org/apps
            session_name: Name for the Telegram session file
            storage_dir: Directory to store downloaded messages
        """
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    async def start(self):
        """Start the Telegram client"""
        await self.client.start()
        
    async def stop(self):
        """Stop the Telegram client"""
        await self.client.disconnect()
        
    async def get_channel_messages(
        self,
        channel_username: str,
        limit: int = 100,
        since_hours: int = 24
    ) -> List[Dict]:
        """
        Get messages from a specific channel
        
        Args:
            channel_username: Username of the channel (without @)
            limit: Maximum number of messages to retrieve
            since_hours: Only get messages from the last N hours
            
        Returns:
            List of messages with their metadata
        """
        messages = []
        since_date = datetime.now() - timedelta(hours=since_hours)
        
        try:
            channel = await self.client.get_entity(channel_username)
            if not isinstance(channel, Channel):
                raise ValueError(f"{channel_username} is not a channel")
                
            async for message in self.client.iter_messages(
                channel,
                limit=limit,
                offset_date=since_date
            ):
                if message.text:  # Only process text messages
                    msg_data = {
                        "channel": channel_username,
                        "message_id": message.id,
                        "date": message.date.isoformat(),
                        "text": message.text,
                        "views": message.views,
                        "forwards": message.forwards
                    }
                    messages.append(msg_data)
                    
        except Exception as e:
            print(f"Error getting messages from {channel_username}: {str(e)}")
            
        return messages
    
    def save_messages(self, messages: List[Dict], filename: str):
        """Save messages to a JSON file"""
        filepath = self.storage_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            
    async def process_channels(
        self,
        channel_usernames: List[str],
        limit_per_channel: int = 100,
        since_hours: int = 24
    ):
        """Process multiple channels and save their messages"""
        all_messages = []
        
        for channel in channel_usernames:
            messages = await self.get_channel_messages(
                channel,
                limit=limit_per_channel,
                since_hours=since_hours
            )
            all_messages.extend(messages)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_messages(all_messages, f"telegram_messages_{timestamp}.json")
        return all_messages

# Example usage:
async def main():
    # These should be stored securely, preferably as environment variables
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    if not API_ID or not API_HASH:
        raise ValueError("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables")
    
    ingestion = TelegramChannelIngestion(API_ID, API_HASH)
    await ingestion.start()
    
    # List of channels to monitor (without @ symbol)
    channels = ["example_channel1", "example_channel2"]
    
    try:
        messages = await ingestion.process_channels(channels)
        print(f"Downloaded {len(messages)} messages from {len(channels)} channels")
    finally:
        await ingestion.stop()

if __name__ == "__main__":
    asyncio.run(main()) 
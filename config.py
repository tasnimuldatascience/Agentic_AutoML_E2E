from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to load environment variables and settings."""
    
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        """Retrieve the OpenAI API key from environment variables."""
        return os.getenv("OPENAI_API_KEY")
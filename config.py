"""
Configuration module for API keys and environment variables.

This module loads environment variables from a .env file and provides
a centralized configuration interface for the entire project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Central configuration class for API keys and settings."""
    
    # API Keys
    XAI_API_KEY = os.getenv('XAI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    
    # API Base URLs
    XAI_BASE_URL = os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1')
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    
    @classmethod
    def validate_api_keys(cls, required_keys=None):
        """
        Validate that required API keys are set.
        
        Args:
            required_keys: List of required key names (e.g., ['XAI_API_KEY', 'GOOGLE_API_KEY'])
                          If None, validates all keys.
        
        Raises:
            ValueError: If any required key is missing.
        """
        if required_keys is None:
            required_keys = ['XAI_API_KEY', 'GOOGLE_API_KEY', 'DEEPSEEK_API_KEY']
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(
                f"Missing required API key(s): {', '.join(missing_keys)}. "
                f"Please set them in your .env file or environment variables."
            )
    
    @classmethod
    def get_xai_config(cls):
        """Get xAI (Grok) API configuration."""
        cls.validate_api_keys(['XAI_API_KEY'])
        return {
            'api_key': cls.XAI_API_KEY,
            'base_url': cls.XAI_BASE_URL
        }
    
    @classmethod
    def get_google_config(cls):
        """Get Google (Gemini) API configuration."""
        cls.validate_api_keys(['GOOGLE_API_KEY'])
        return {
            'api_key': cls.GOOGLE_API_KEY
        }
    
    @classmethod
    def get_deepseek_config(cls):
        """Get DeepSeek API configuration."""
        cls.validate_api_keys(['DEEPSEEK_API_KEY'])
        return {
            'api_key': cls.DEEPSEEK_API_KEY,
            'base_url': cls.DEEPSEEK_BASE_URL
        }


# Convenience functions for backward compatibility
def get_xai_api_key():
    """Get xAI API key."""
    return Config.XAI_API_KEY

def get_google_api_key():
    """Get Google API key."""
    return Config.GOOGLE_API_KEY

def get_deepseek_api_key():
    """Get DeepSeek API key."""
    return Config.DEEPSEEK_API_KEY

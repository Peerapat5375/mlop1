"""
Unit Tests for Data Preprocessing Functions
-----------------------------------------
"""
import sys
import os
import pandas as pd
import pytest
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.01_data_preprocessing import clean_text

def test_clean_text_basic():
    """Test basic text cleaning functionality"""
    # Test case 1: Basic cleaning
    text = "Hello! This is a #test @mention 123"
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)
    assert '#' not in cleaned
    assert '@' not in cleaned
    
def test_clean_text_empty():
    """Test cleaning empty or None input"""
    # Test case 2: Empty string
    assert clean_text('') == ''
    
    # Test case 3: None input should return empty string
    assert clean_text(None) == ''

def test_clean_text_special_chars():
    """Test cleaning text with special characters"""
    # Test case 4: Special characters
    text = "!@#$%^&*()_+ Hello 123"
    cleaned = clean_text(text)
    assert any(char not in cleaned for char in "!@#$%^&*()_+")
    assert 'hello' in cleaned.lower()

def test_clean_text_urls():
    """Test cleaning text with URLs"""
    # Test case 5: URLs
    text = "Check this link http://example.com and https://test.com"
    cleaned = clean_text(text)
    assert 'http' not in cleaned.lower()
    assert 'www' not in cleaned.lower()

def test_clean_text_hashtags():
    """Test cleaning text with hashtags"""
    # Test case 6: Hashtags
    text = "#awesome #python #test"
    cleaned = clean_text(text)
    assert '#' not in cleaned
    assert 'awesome' in cleaned.lower()
    assert 'python' in cleaned.lower()
    assert 'test' in cleaned.lower()

def test_clean_text_numbers():
    """Test cleaning text with numbers"""
    # Test case 7: Numbers
    text = "I am 123 years old"
    cleaned = clean_text(text)
    assert '123' not in cleaned
    assert 'years old' in cleaned.lower()

def test_clean_text_mixed_case():
    """Test cleaning text with mixed case"""
    # Test case 8: Mixed case
    text = "HeLLo WoRLD"
    cleaned = clean_text(text)
    assert cleaned.lower() == cleaned

def test_clean_text_multiple_spaces():
    """Test cleaning text with multiple spaces"""
    # Test case 9: Multiple spaces
    text = "hello    world   test"
    cleaned = clean_text(text)
    assert "    " not in cleaned
    assert "   " not in cleaned
    assert cleaned.count(' ') == 2

def test_clean_text_emoji():
    """Test cleaning text with emoji"""
    # Test case 10: Emoji
    text = "Hello 👋 World 🌍"
    cleaned = clean_text(text)
    assert '👋' not in cleaned
    assert '🌍' not in cleaned
    assert 'hello' in cleaned.lower()
    assert 'world' in cleaned.lower()
"""
Pytest Configuration File
------------------------
"""
import pytest
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture
def sample_tweets():
    """Fixture providing sample tweets for testing"""
    return [
        "I hate you because of your religion! #hate",
        "You are such an idiot! @mention",
        "Women are weak and can't lead 😠",
        "My grandma is so bad with technology http://test.com",
        "I love my friends from all countries ❤️",
        "Old people should stop using phones",
        "You are amazing!",
    ]

@pytest.fixture
def sample_labels():
    """Fixture providing sample labels for testing"""
    return [2, 1, 1, 3, 0, 3, 0]  # Corresponding to the sample_tweets
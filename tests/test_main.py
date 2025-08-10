"""
Tests for the main module.
"""
from src.main import greet

def test_greet():
    """Test the greet function."""
    # Test with a regular name
    assert greet("Alice") == "Hello, Alice! Welcome to the project."
    
    # Test with an empty string
    assert greet("") == "Hello, ! Welcome to the project." 
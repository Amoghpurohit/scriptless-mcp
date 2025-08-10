"""
Main module containing core functionality.
"""

def greet(name: str) -> str:
    """
    Returns a greeting message for the given name.
    
    Args:
        name (str): Name of the person to greet
        
    Returns:
        str: Greeting message
    """
    return f"Hello, {name}! Welcome to the project."

if __name__ == "__main__":
    print(greet("User")) 
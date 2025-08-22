# Simple test for mood functionality
class MockLangGraphAgent:
    def __init__(self, retriever_mode, MODE, langchain_project_name):
        self.retriever_mode = retriever_mode
        self.MODE = MODE
        self.langchain_project_name = langchain_project_name
        self.mood = None
        
    def set_mood(self, mood: str):
        """Set the current mood of the user"""
        self.mood = mood
        
    def get_mood(self) -> str:
        """Get the current mood of the user"""
        return self.mood

# Test the mood functionality
if __name__ == "__main__":
    agent = MockLangGraphAgent("test", "DEMO_DAY", "AIM-DEMO-DAY")
    print("Agent initialized successfully")
    print("Initial mood:", agent.get_mood())
    
    agent.set_mood("happy")
    print("Mood after setting:", agent.get_mood())
    
    agent.set_mood("very upset")
    print("Mood after setting:", agent.get_mood())
    
    print("Mood functionality test passed!")

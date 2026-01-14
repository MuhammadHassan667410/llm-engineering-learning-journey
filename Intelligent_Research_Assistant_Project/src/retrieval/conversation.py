from typing import List, Dict, Any

class ConversationManager:
    def __init__(self, max_history: int = 10):
        """
        Manages the chat history for the AI assistant.
        
        Args:
            max_history (int): The maximum number of messages to keep in memory.
                               Keeping this low (e.g., 10) prevents the context 
                               from getting too large and expensive.
        """
        # This list acts as our storage. 
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_user_message(self, message: str):
        """
        Save what the human said.
        """
        self.history.append({"role": "user", "content": message})
        self._trim_history()

    def add_assistant_message(self, message: str):
        """
        Save what the AI replied.
        """
        self.history.append({"role": "assistant", "content": message})
        self._trim_history()

    def _trim_history(self):
        """
        Internal function to keep the memory size in check.
        If we have more than 'max_history' messages, remove the oldest ones.
        """
        if len(self.history) > self.max_history:
            # Slicing [-N:] keeps only the last N items
            self.history = self.history[-self.max_history:]

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the raw list of messages.
        This is what 'prompts.py' needs to build the final prompt.
        """
        return self.history

    def get_formatted_history(self) -> str:
        """
        Return a string version for debugging or simple display.
        """
        formatted = ""
        for msg in self.history:
            role = "Human" if msg['role'] == 'user' else "AI"
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def clear(self):
        """
        Wipe the memory clean. Used when the user clicks 'New Chat'.
        """
        self.history = []
        print("Conversation history cleared.")

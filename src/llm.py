"""
Minimal implementation of the llm_handler module to support GraphRAG
"""

from langchain_openai import ChatOpenAI


class ChatOpenAITool(ChatOpenAI):
    """
    Wrapper for LLM models that is compatible with LangChain.

    This class inherits from langchain_openai.ChatOpenAI to ensure
    full compatibility with the LangChain ecosystem.
    """

    def __init__(self, model: str, **kwargs):
        # Pass all arguments to the parent class constructor.
        # This ensures that the model is initialized correctly and
        # is ready to be used by LangChain components.
        super().__init__(model=model, **kwargs)

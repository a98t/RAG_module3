"""
LLM Client Module

This module provides functions to interact with OpenAI's Chat Completion API
for generating natural language responses.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0  # Deterministic responses for factual Q&A
DEFAULT_MAX_TOKENS = 512


def ask_llm(
    prompt: str,
    system_message: str = "You are a helpful assistant expert on the Premier League.",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """
    Send a prompt to the LLM and get a response.
    
    Args:
        prompt: The user's question or input
        system_message: The system prompt that sets the AI's behavior
        temperature: Controls randomness (0=deterministic, 1=creative)
        max_tokens: Maximum length of the response
        
    Returns:
        The LLM's response as a string
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


def ask_llm_with_context(
    question: str,
    context_documents: List[Dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """
    Ask the LLM a question with relevant context documents.
    This is the core RAG generation step.
    
    Args:
        question: The user's question
        context_documents: List of dicts with 'title' and 'content' keys
        temperature: Controls randomness
        max_tokens: Maximum response length
        
    Returns:
        The LLM's answer based on the provided context
    """
    # Build the context string from documents
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        context_parts.append(f"Document {i}: {doc['title']}\n{doc['content']}")
    
    context_string = "\n\n---\n\n".join(context_parts)
    
    # Create the RAG prompt
    system_message = """You are an expert assistant on the English Premier League.

Your task is to answer questions based ONLY on the provided context documents.

Important rules:
1. Use ONLY information from the context documents provided
2. Do not use your general knowledge about the Premier League
3. If the answer is not in the context, say "I don't have enough information in the provided context to answer this question."
4. Cite specific details from the context (names, numbers, dates)
5. Be concise and factual
6. Format your answer in clear paragraphs or bullet points"""

    prompt = f"""Context Documents:

{context_string}

Question: {question}

Answer:"""

    return ask_llm(
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens
    )


def get_model_info() -> Dict[str, any]:
    """
    Returns information about the configured LLM model.
    
    Returns:
        Dictionary with model configuration details
    """
    return {
        "model": LLM_MODEL,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_max_tokens": DEFAULT_MAX_TOKENS
    }


if __name__ == "__main__":
    # Test the LLM client
    print("Testing LLM Client...")
    print(f"Model: {LLM_MODEL}\n")
    
    # Test basic question
    test_question = "What is the Premier League?"
    print(f"Test question: {test_question}")
    
    try:
        answer = ask_llm(test_question)
        print(f"✅ Response received:")
        print(f"{answer}\n")
        
        # Test with context
        print("Testing RAG-style question with context...")
        test_docs = [
            {
                "title": "The Invincibles",
                "content": "Arsenal's 2003–04 team, nicknamed 'The Invincibles,' completed the entire Premier League season unbeaten. They recorded 26 wins and 12 draws, finishing with 90 points and a +47 goal difference."
            }
        ]
        
        context_question = "How many points did Arsenal get in their unbeaten season?"
        print(f"Question: {context_question}")
        
        context_answer = ask_llm_with_context(context_question, test_docs)
        print(f"✅ Context-based response:")
        print(f"{context_answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

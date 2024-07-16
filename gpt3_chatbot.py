"""
Module: gpt3_chatbot.py



Dependencies:
- openai: OpenAI's Python library for API interaction.
- dotenv: To load environment variables from a .env file.

Usage:
1. Ensure environment variable OPENAI_API_KEY is set with your OpenAI API key.
2. Instantiate GPT3Assistant.
3. Use the g_chat method to interact with the assistant by providing user input.
"""

import openai
from dotenv import load_dotenv
import os
import streamlit as st

class GPT3Assistant:
    """
    A class to interact with OpenAI's GPT-3.5 API for creating a chatbot assistant.
    """

    def __init__(self):
        """
        Initialize the GPT3Assistant with API key from environment variables.
        """
        load_dotenv()
        openai.api_key = st.secrets["OPENAI_API_KEY"]

    def get_openai_response(self, messages):
        """
        Get response from OpenAI GPT-3.5 API based on provided chat messages.

        Args:
        - messages (list): List of dictionaries containing chat messages with roles.

        Returns:
        - str: Generated response from OpenAI.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
        )

        return response['choices'][0]['message']['content']

    def g_chat(self, msg):
        """
        Conduct a chat with the assistant using OpenAI's GPT-3.5.

        Args:
        - msg (str): User input message.

        Returns:
        - str: Response from the assistant.
        """
        input_msg = msg
        chat_messages = [
            {'role': 'system', 'content': 'You are Clinton, a helpful assistant.'},
            {'role': 'user', 'content': input_msg}
        ]
        response = self.get_openai_response(chat_messages)
        
        return response

# Example usage
if __name__ == "__main__":
    assistant = GPT3Assistant()
    user_input = input("Enter a message: ")
    response = assistant.g_chat(user_input)
    print(response)

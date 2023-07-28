import streamlit as st 
from streamlit_chat import message 
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import os 


def main():
    if os.getenv("OPEN_API_KEY") is None or os.getenv("OPEN_API_KEY"):
        print("Key is not set")
        exit(1)
    else:
        print("Key is set")

    chat = ChatOpenAI()

    messages = SystemMessage(content="You are a helpful assistant.")

    with st.sidebar:
        user_input =  st.text_input("You message: ", key = "user_input")

    if user_input:
        message(user_input, is_user=True)
        messages.append(HumanMessage(content=user_input))
        response = chat(message)
        message(response.content, is_user=False)

if __name__ == '__main__':
    main()
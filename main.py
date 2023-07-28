import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.sql_database import SQLDatabase
from langchain.embeddings import OpenAIEmbeddings

def main():

    chat = ChatOpenAI()
    

    messages = [SystemMessage(content="You are a helpful assistant.")]

    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant. ")]
    
    
    with st.sidebar:
        user_input =  st.text_input("You message: ", key = "user_input")

    if user_input:
        message(user_input, is_user=True)
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking"):
            response=chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        message(response.content, is_user=False)


    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_AI')
if __name__ == '__main__':
    main()
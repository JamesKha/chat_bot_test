import os
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message
from langchain.agents import load_tools, initialize_agent, AgentType, create_pandas_dataframe_agent
from langchain.sql_database import SQLDatabase
from langchain.embeddings import OpenAIEmbeddings
import numpy as np 


def main():
    df = pd.read_csv('MOCK_DATA.csv')
    agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)

    user_count, ai_count = 0, 0
    
    embeddings = OpenAIEmbeddings()
    
    if "initialized" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant. ")]
        #st.session_state.embeddings = np.empty((0, 768))
        st.session_state.embeddings = None
        st.session_state.initialized = True
    
    with st.sidebar:
        user_input =  st.text_input("You message: ", key = "user_input")


    messages = st.session_state.messages
    for msg in messages:
        if isinstance(msg,HumanMessage):
            message(msg.content, is_user=True, key=str(user_count) + '_user')
            user_count+= 1
        elif isinstance(msg,AIMessage):
            message(msg.content, is_user=False, key=str(ai_count) + '_AI')
            ai_count += 1


    if user_input:
        message(user_input, is_user=True)
        st.session_state.messages[-1] = HumanMessage(content=user_input)
        
        with st.spinner("Thinking"):
            response=agent(st.session_state.messages)
        st.session_state.messages[-1] = AIMessage(content=response.__str__())
        message(response['output'].__str__(), is_user=False)


    user_embedding = embeddings.embed_query(user_input)
    ai_embedding = embeddings.embed_query(response['output'].__str__())


        
if __name__ == '__main__':
    main()
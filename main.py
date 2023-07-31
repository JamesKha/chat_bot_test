import os
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message
from langchain.agents import load_tools, initialize_agent, AgentType, create_pandas_dataframe_agent
from langchain.embeddings import OpenAIEmbeddings
import numpy as np 


def main():
   
    
    if "initialized" not in st.session_state:
        st.session_state.messages = None
        #st.session_state.embeddings = np.empty((0, 768))
        st.session_state.embeddings = None
        st.session_state.initialized = True
    
    with st.sidebar:
        user_input =  st.text_input("Your message: ", key = "user_input")
        uploaded_file = st.file_uploader("Choose a file: ")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)

        user_count, ai_count = 0, 0
    
        embeddings = OpenAIEmbeddings()

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


            #user_embedding = embeddings.embed_query(user_input)
            #ai_embedding = embeddings.embed_query(response['output'].__str__())
            #print(type(user_embedding), type(ai_embedding))
        
            #if st.session_state.embeddings is None:
                #st.session_state.embeddings = np.empty((0, user_embedding.shape[1]))

            #if user_embedding.shape[1] != st.session_state.embeddings.shape[1]:
                #user_embedding = np.zeros((user_embedding.shape[0], st.session_state.embeddings.shape[1]))
                #user_embedding[:, :st.session_state.embeddings.shape[1]] = embeddings
        
            #if ai_embedding.shape[1] != st.session_state.embeddings.shape[1]:
                #ai_embedding = np.zeros((ai_embedding.shape[0], st.session_state.embeddings.shape[1]))
                #ai_embedding[:, :st.session_state.embeddings.shape[1]] = embeddings


            #st.session_state.embeddings = np.vstack([st.session_state.embeddings, user_embedding, ai_embedding])

        
if __name__ == '__main__':
    main()
import os
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message
from langchain.agents import load_tools, initialize_agent, AgentType, create_pandas_dataframe_agent
from langchain.embeddings import OpenAIEmbeddings
import numpy as np 

max_embeds = 25 
def main():
   
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if "initialized" not in st.session_state:
        st.session_state.messages = ["How are you?"]
        st.session_state.initialized = True
    
    with st.sidebar:
        user_input =  st.text_input("Your message: ", key = "user_input")
        uploaded_file = st.file_uploader("Choose a file: ")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        try:
            st.session_state.embeddings = np.load("embeddings.npy")
        except:
            st.session_state.embeddings = np.empty((0, 768))
        
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True, memory = st.session_state.embeddings)

        user_count, ai_count = 0, 0
    
        embeddings = OpenAIEmbeddings()

        messages = st.session_state.messages
       
        if user_input:
            message(user_input, is_user=True)
            st.session_state.messages[-1] = HumanMessage(content=user_input)
        
            with st.spinner("Thinking"):
                response=agent(st.session_state.messages)
            st.session_state.messages[-1] = AIMessage(content=response.__str__())
            

            for msg in messages:
                if isinstance(msg,HumanMessage):
                    message(msg.content, is_user=True, key=str(user_count) + '_user')
                    user_count+= 1
                elif isinstance(msg,AIMessage):
                    message(response['output'].__str__(), is_user=False, key=str(ai_count) + '_AI')
                    ai_count += 1
            user_embedding = np.array(embeddings.embed_query(user_input))
            ai_embedding = np.array(embeddings.embed_query(response['output'].__str__()))

            if st.session_state.embeddings is None:
                st.session_state.embeddings = np.empty((0, user_embedding.shape[0]))

            if user_embedding.shape[0] != st.session_state.embeddings.shape[1]:
                user_embedding = np.zeros((user_embedding.shape[0], st.session_state.embeddings.shape[1]))
                user_embedding[:, :st.session_state.embeddings.shape[1]] = embeddings
        
            if ai_embedding.shape[0] != st.session_state.embeddings.shape[1]:
                ai_embedding = np.zeros((ai_embedding.shape[0], st.session_state.embeddings.shape[1]))
                ai_embedding[:, :st.session_state.embeddings.shape[1]] = embeddings


            st.session_state.embeddings = np.vstack([st.session_state.embeddings[-max_embeds:], user_embedding, ai_embedding])
            np.save('embeddings.npy', st.session_state.embeddings)
        
if __name__ == '__main__':
    main()
import streamlit as st 
from streamlit_chat import message 


def main():


    message("Hello how are you")

    message("I'm good", is_user=True)



    with st.sidebar:
        user_input =  st.text_input("You message: ", key = "user_input")


if __name__ == '__main__':
    main()
import streamlit as st
from src.agent_setup import *

def main():
    '''setup our Streamlit'''
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")
    agent = profile_agent()

    st.header("AI research agent")
    query = st.text_input("What would you like to know?")

    if query:
        st.write("Doing research for ", query)
        result = agent({"input" : query})
        st.info(result['output'])


if __name__=="__main__":
    main()
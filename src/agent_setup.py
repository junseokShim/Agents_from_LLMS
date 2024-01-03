import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv('./env/api.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API = os.getenv("SERPER_API_KEY")
BROWSERLESS_API = os.getenv("BROSERLESS_API_KEY")


def search_tool(query):
    '''this function help our agent does browse through the internet, 
    using Serper(low-cost Google Search API)
    args: query
    return: response(from google)
    '''
    base_url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY':SERPER_API,
        'Content-Type':'application/json'
    }
    response = requests.request("POST", base_url,
                                headers=headers,
                                data=payload)
    return response.text


def summary(objective, content):
    '''summaryze text from soup.get_text() method using OpneAI(GPT)
    args : 
        - objective : 
        - content : 
    return : 
    '''
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    text_spliter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 1000, chunk_overlap=500)
    
    docs = text_spliter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective} : 
    '{text}'
    SUMMARY : 
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text","objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    return summary_chain.run(input_documents=docs, objective=objective)
    
def scrape_website(objective: str, url:str):
    '''this function helps our agent do open website(using Browserless) and 
    find what you're looking for automatically(using BeautifulSoup).
    args: 
        - objective : query
        - url : url
    return: text or summarization result from url
    '''
    print("Scraping website...")
    # Define the headers for the request(from search_tool)
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type' : 'application/json'
    } 

    # Define the data to be sent in the request(from search_tool)
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={BROWSERLESS_API}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        # Write validation
        if len(text)>1000:
            # Call summary function to summaryze text
            return summary(objective, text)
        
        return text

    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return
    

class ScrapeWebsiteInput(BaseModel):
    '''Inputs for scrape_website'''
    objective: str = Field(
        description = "The objective & task that users give to the agent"
    )
    url: str = Field(
        description = "The url of website to be scraped"
    )


class ScrapeWebsiteTool(BaseTool):
    '''define names and desc of scarpe tool and include scrape_website execution method'''
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        '''execute the scrape_website method
        '''
        raise scrape_website(objective, url)

    def _arun(self, url: str):
        '''This method indicates that the feature has not yet been implemented to user.
        '''
        raise NotImplementedError("error here")
    

def profile_agent():
    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
    content="""
        You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
        you do not make things up, you will try as hard as possible to gather facts & data to back up the research

        Please make sure you complete the objective above with the following rules:
        1/ You should do enough research to gather as much information as possible about the objective
        2/ If there are url of relevant links & articles, you will scrape it to gather more information
        3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
        4/ You should not make things up, you should only write facts & data that you have gathered
        5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        """
    )

    agent_kwargs = {
        "extra_prompt_messages" : [MessagesPlaceholder(variable_name="memory")],
        "system_message" : system_message,
    }


    ### initialize our agent
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    memory = ConversationSummaryBufferMemory(
        memory_key = "memory",
        return_messages = True,
        llm = llm,
        max_token_limit = 1000
    )

    agent = initialize_agent(
        tools,
        llm,
        agent = AgentType.OPENAI_FUNCTIONS,
        verbose = True,
        agent_kwargs = agent_kwargs,
        memory = memory
    )

    return agent
    
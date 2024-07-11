import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyJSONLoader
from langchain_community.document_loaders import JSONLoader
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader




with open('travel_data.json', 'r', encoding='utf-8') as file:
    travel_data = json.load(file)

file_path = 'travel_data.json'


# Load the GROQ and OpenAI API keys
groq_api_key = "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" #os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = "AIzaSyDtDfEErK6bZMJV-EZq2vZkrluXuKuOSB0"#os.getenv("GOOGLE_API_KEY")
# google_api_key = "AIzaSyDtDfEErK6bZMJV-EZq2vZkrluXuKuOSB0"
# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")




loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',  # Adjust the jq_schema as per your JSON structure
    text_content=False  # Set to True if your JSON contains text content to be processed
)

# Load documents
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

vectors = FAISS.from_documents(final_documents, embeddings)




# Streamlit UI
st.title('Travel Itinerary Recommender')

# User input for query
user_query = st.text_input('Enter your travel preferences or needs:')

if user_query:
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the response based on the question. output should be in the format
        description: places to visit
        activities: activities to do in that location
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Perform retrieval and display results
    # Perform retrieval and display results
    response = retrieval_chain.invoke({'input': user_query})
    st.subheader('Top Recommendations:')
    # st.write(response)
    # st.write(response['answer'])


    st.write("............................")
    st.write(response['answer'])
    st.write("............................")

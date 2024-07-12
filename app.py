import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import JSONLoader


load_dotenv()

with open('travel_data.json', 'r', encoding='utf-8') as file:
    travel_data = json.load(file)

file_path = 'travel_data.json'
# Load the GROQ and OpenAI API keys
groq_api_key =  os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',  # jq_schema as per your JSON structure
    text_content=False
)

# Load JSON
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)
image_path = "img.png"
# Streamlit UI
st.title('Trip Explorer')
st.image(image_path, use_column_width=True)
# User input for query
user_query = st.text_input('Enter your travel preferences or needs:')
if user_query:
    # Defining the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the response based on the question. output should be in the following format each representing in a new line.
        If the no of recommendations is more than 3 for the same place, make sure you give top 3 recommendations of the destinationb asked else give the no of recommendations available. 
        description: places to visit
        activities: activities to do in that location
        duration: no of days it takes
        
        Handle typos and variations in questions asked.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # Creating document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Perform retrieval and display results
    response = retrieval_chain.invoke({'input': user_query})
    st.subheader('Top Recommendations:')
    st.write("............................")
    st.write(response['answer'])
    st.write("............................")

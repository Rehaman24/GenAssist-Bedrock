from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings  # Import BedrockEmbeddings module
import boto3
import os
import streamlit as st

# Set AWS profile
os.environ["AWS_PROFILE"] = "IAM user"

# Initialize AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define Bedrock embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Initialize Bedrock language model
modelID = "anthropic.claude-v2"
llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)

# Define chatbot function
def my_chatbot(language, freeform_text, bedrock_embeddings):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})
    return response

# Function for vector store creation
def get_vector_store(docs, bedrock_embeddings):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

# Function for data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("path to pdf files")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Main Streamlit app
def main():
    st.set_page_config("GenAssitChat")
    st.header("GenAssist ChatBot")
    user_question = st.text_input("Hit a query or from document")
    
    # Sidebar for updating vector store
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vectorstore_faiss = get_vector_store(docs, bedrock_embeddings)
                st.success("Done")

    # Buttons for different chatbot outputs
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            st.write(my_chatbot("english", user_question, bedrock_embeddings)['text'])
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            st.write(my_chatbot("english", user_question, bedrock_embeddings)['text'])
            st.success("Done")

# Start the Streamlit app
if __name__ == "__main__":
    main()

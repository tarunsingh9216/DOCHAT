""""
1) Read the PDF and Question
2) Data Cleaning
3) Splitting
4) Tokenization
5) Convert the content into vector embeddings
6) Use a vector DB to store and read the vector embeddings
7) Import the LLM(ollama)
8) Quantize and finetune the LLM
9) Use the LLM for generate answers from a contex.
"""

import re
import os
import torch
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

load_dotenv()   
pinecone_key = os.getenv("PINECONE_API_KEY")


def main():

    TEXT = ""

    st.set_page_config("DOCHAT")
    st.header("Chat With PDF using LLM💁")

    uploded_file = st.file_uploader("Uplode a PDF file", type="pdf")

    if uploded_file is not None:
        st.write("Extracting the Text form the Pdf")
        save_dir = 'D:\personal\DocChat\data'
        save_path = os.path.join(save_dir, uploded_file.name)

        if os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploded_file.getbuffer())
                st.success("File saved")
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploded_file.getbuffer())
                st.success("Path Created File Saved")
        TEXT = read_the_PDF()
        
        st.success("Text Extraction Compelted")

    user_question = st.text_input("Enter You Question")
    button = st.button("Enter")

    clean_txt = data_cleaning(TEXT)
    vectors = create_vectorsStore(clean_txt, user_question)
    final_prompt = prompt_template(user_question, vectors)
    answer = LLM_openSource(final_prompt)

    if button is True:
        if uploded_file is not None:
            st.write("Answer : ", final_prompt)           
        else:
            st.write("Pless Uplode the File.")




def read_the_PDF():

    doc = PdfReader('D:\personal\DocChat\data\Introduction to Machine Learning.pdf')
    text_content = ""
    for page in doc.pages:
        text_content += page.extract_text()

    return text_content



def data_cleaning(text):
    """"
    - Lowercasing
    - Removing special characters, punctuation, and numbers
    - Removing extra spaces
    - Removing stopwords
    - Spliting the Text

    """

    # Data Cleaning
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()

    # Data Splitting
    doc = Document(page_content=cleaned_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_chunks = text_splitter.split_documents([doc])

    return texts_chunks 


def create_vectorsStore(doc, query):

    INDEX_NAME = "dochat"
    
    vector_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = PineconeVectorStore.from_documents(doc, embedding=vector_model, index_name=INDEX_NAME)
    
    # query = "what is machine learning"
    similar_docs = vector_db.similarity_search(query)

    return similar_docs
    

def prompt_template(query,retrieved_text):

    prompt_temp = """"

    You are an AI assistant that answers questions based on the provided document context. 
    Use only the given information to generate accurate responses.

    User Query:
    {query}

    Context from the document:
    {retrieved_text}
    

    Important Rules:
    1. Only use the provided document context to answer.
    2. If the answer is not found in the context, say "I don't know."
    3. Keep the answer clear and to the point.

    Your Answer:
    
    
    """
    PROMPT = PromptTemplate(
        template=prompt_temp, input_variables=["query", "retrieved_text"]
    )
    formated_prompt = PROMPT.format(query = query, retrieved_text = retrieved_text)

    return formated_prompt


def LLM_openSource(prompt):
    
    MODEL_NAME = "Tarun9216/Llama-2-7b-DoChat-finetune"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map = "auto",
        torch_dtype=torch.float16   
    )

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")

    return result



if __name__ == "__main__":
    main()

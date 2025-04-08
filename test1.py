import re
import os
import spacy
import ollama
import chromadb
from pinecone import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore






# Load spaCy model
# nlp = spacy.load("en_core_web_sm")


os.environ['PINECONE_API_KEY'] = 'pcsk_4hVVbm_C1Nb5pKQySdMQ7dhUBqxZfH1Uam212xNJWbKueLzBrXnUtK4Zex5DA3Dm48anoL'

def clean_text(text):
    """
    Cleans text by performing:
    - Lowercasing
    - Removing special characters, punctuation, and numbers
    - Removing extra spaces
    - Removing stopwords
    - Lemmatization
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)  # Keeps only alphabets and spaces
    
    # Process text with spaCy
    # doc = nlp(text)
    
    # Lemmatization & Stopword Removal

    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    return cleaned_text


def splitting(doc):
    document = Document(page_content=doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    texts = text_splitter.split_documents([document])
    return texts


def data_ingestion():
    loader=PyPDFDirectoryLoader("D:\personal\RAG project\data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
    
    docs=text_splitter.split_documents(documents)
    return docs

def token(doc):
    # model_name = "llama2"
    # tokens = ollama.tokenize(model=model_name, text=text)
    # print(tokens)
    # embedding_model = OllamaEmbeddings(model='llama2')
    # vector = embedding_model.embed_documents(doc)
    # print("Vectors", vector[:5])
    # print("vector lenght", len(vector))

    index_name = "dochat"

    # text_list = [d.page_content for d in doc]

    # print(text_list)


    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # em = embeddings.embed_documents(text_list)
    
    # db = FAISS.from_documents(doc, embedding=embeddings)

    vector_store = PineconeVectorStore.from_documents(doc, embedding=embeddings, index_name=index_name)

    # query = ""



    # print(len(em))
    # print(len(em[0]))
    # print(em[0][:5])

    print("null")



# Example usage

if __name__ == "__main__":
    sample_text = "Hello!! This is an example text, with numbers like 123 and special characters ***!!!"
    cleaned_output = clean_text(sample_text)
    split_text = splitting(cleaned_output)
    # print(cleaned_output)
    # print(split_text)
    doc = data_ingestion()
    print(doc)
    token(doc)

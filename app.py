import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationalBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


# creating function for pdf to raw data
def get_pdf_text(pdf_docs):
    """
    This is the function for data from multiple PDFs
    """
    # creating empty string variable to store content of PDFs
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    This is the function for reating data chunks with help of langchain model CharacteTextSplitter
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory =  ConversationalBufferMemory(memory_key = 'chat_hostory', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat With Multiple PDFs'
                       ,page_icon=':books:')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    st.header('Chat with multiple PDFs :books:')
    st.text_input('Ask a question about your documents:')

    with st.sidebar:
        st.subheader("Your Document's")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process",
            accept_multiple_files=True
            )
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store(Embedings)
                vectorestore = get_vectorstore(text_chunks)
                
                # Note: we can you OpenAI Embeding models for Embedings if you are doing it for any organization 
                # But We will Open source Embedings (Instructor Finetuned Text Embedings)
                
                # Create conversation chains
                st.session_state.conversation = get_conversation_chain(vectorestore)
                

if __name__ == '__main__':
    main() 

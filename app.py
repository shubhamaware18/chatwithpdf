import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# creating function for pdf to raw data
def get_pdf_text(pdf_docs):
    # creating empty string variable to store content of PDFs
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def main():
    load_dotenv()
    st.set_page_config(page_title='Chat With Multiple PDFs',page_icon=':books:')

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
                st.write(raw_text)
                # get the text chunks

                # create vectore store

if __name__ == '__main__':
    main() 

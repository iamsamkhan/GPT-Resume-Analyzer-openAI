import time
import numpy as np
import pandas as pd
import streamlit as st
import os
import io
import pdf2image
import base64
from PIL import Image
from streamlit_option_menu import option_menu
#from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
#from langchain.chains.question_answering import load_qa_chain

import warnings
warnings.filterwarnings('ignore')


# class resume_analyzer:

#     def pdf_to_chunks(pdf):
#         # read pdf and it returns memory address
#         pdf_reader = PdfReader(pdf)

#         # extrat text from each page separately
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         # Split the long text into small chunks.
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=700,
#             chunk_overlap=200,
#             length_function=len)

#         chunks = text_splitter.split_text(text=text)
#         return chunks
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_gpt_response(openai_api_key, chunks, analyze):

        # Using OpenAI service for embedding
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Facebook AI Similarity Serach library help us to convert text data to numerical vector
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

        # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
        docs = vectorstores.similarity_search(query=analyze, k=3)

        # creates an OpenAI object, using the ChatGPT 3.5 Turbo model
        llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=openai_api_key)

        # question-answering (QA) pipeline, making use of the load_qa_chain function
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=analyze)
        return response

def get_gpt_response(input,pdf_content,prompt):
    model=OPENAI_API_KEY.ChatOpenAI('gpt-3.5-turbo')
    response=model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

# def openai(openai_api_key, chunks, analyze):

#         # Using OpenAI service for embedding
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#         # Facebook AI Similarity Serach library help us to convert text data to numerical vector
#         vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

#         # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
#         #docs = vectorstores.similarity_search(query=analyze, k=3)

#         # creates an OpenAI object, using the ChatGPT 3.5 Turbo model
#         llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=openai_api_key)

#         # question-answering (QA) pipeline, making use of the load_qa_chain function
#         chain = load_qa_chain(llm=llm, chain_type='stuff')

#         response = chain.run(input_documents=docs, question=analyze)
#         return response

#second apporach 
# def get_summary_for_resume(description):
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt="Summarize the following in bullet points for my resume \n\n"+description,
#         temperature=0,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     if len(response['choices'][0]) > 0:
#         points = response['choices'][0]['text'].split("\n")
#         s = ""
#         # Remove the Bullet point from the response text
#         for point in points:
#             s += point[1:]+"\n"
#         return s
#     return ""

# def get_summary_for_projects(description):
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt="Summarize the following in 2 bullet points for my resume \n\n"+description,
#         temperature=0,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     if len(response['choices'][0]) > 0:
#         points = response['choices'][0]['text'].split("\n")
#         s = ""
#         # Remove the Bullet point from the response text
#         for point in points:
#             s += point[1:]+"\n"
#         return s
#     return ""
# def get_gpt_response(prompt, max_tokens=200):
#     prompt = f"Answer the following question: {prompt}\nAnswer:"
#     response = openai.Completion.create(
#         engine="davinci",
#         prompt=prompt,
#         max_tokens=500,
#         n=1,
#         stop="\n",  # Set a stop sequence here
#         temperature=0.7,
#     )
#     return response

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        # Take the first page for simplicity, or loop through images for all pages
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App



st.set_page_config(page_title="Resume Expert")

st.header("JobFit Analyzer")
st.subheader('This Application helps you in your Resume Review with help of CAHTGPT  AI [LLM]')
input_text = st.text_input("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your Resume(PDF)...", type=["pdf"])
pdf_content = ""

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")

submit2 = st.button("How Can I Improvise my Skills")

submit3 = st.button("What are the Keywords That are Missing")

submit4 = st.button("Percentage match")

input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt2 = """
You are an Technical Human Resource Manager with expertise in data science, 
your role is to scrutinize the resume in light of the job description provided. 
Share your insights on the candidate's suitability for the role from an HR perspective. 
Additionally, offer advice on enhancing the candidate's skills and identify areas where improvement is needed.
"""

input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. As a Human Resource manager,
 assess the compatibility of the resume with the role. Give me what are the keywords that are missing
 Also, provide recommendations for enhancing the candidate's skills and identify which areas require further development.
"""
input_prompt4 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gpt_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gpt_response(input_prompt2, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gpt_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit4:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gpt_response(input_prompt4, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")


st.markdown("---")
st.caption("Resume Expert - Making Job Applications Easier")
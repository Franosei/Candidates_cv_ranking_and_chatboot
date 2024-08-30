import os
import sys
from PyPDF2 import PdfWriter, PdfReader
import PyPDF2
import json
import streamlit as st
import pandas as pd
from docx import Document
import time
from openai import OpenAI
import tiktoken
import openai
from dotenv import load_dotenv
from streamlit_extras.metric_cards import style_metric_cards

load_dotenv()
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image("rank.png",width=100)
with col3:
    st.write(' ')

def llm(text, system_message, delimiter="####", print_response=False, retries=3, sleep_time=10):
    while retries > 0:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}{text}{delimiter}"}
        ]
        max_tokens = 16000 - (len(text) + len(system_message)) - 13
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0
            )
            response = completion.choices[0].message.content
            if print_response:
                print(response)
            break
        except openai.RateLimitError as e:
            print('Catching RateLimitError, retrying in 1 minute...')
            retries -= 1
            time.sleep(sleep_time)
    return response

def process_files(job_spec_files, resume_files):
    # Processing logic for job specification
    combined_text = []
    if job_spec_files:
        for job_spec_file in job_spec_files:
            pdf_reader = PdfReader(job_spec_file)
            for page_num in range(len(pdf_reader.pages)):
                combined_text.append(pdf_reader.pages[page_num].extract_text())
    job_descrip = '\n'.join(combined_text)
    system_message = "I'm providing a job advertisement along with scores for each category.\
        Please rewrite the provided job advertisement along with the exact scores for each category."
    descrip_score = llm(job_descrip, system_message, delimiter="####", print_response=True, retries=3, sleep_time=10)

    # Processing logic for candidate resumes
    candidates_scores = []
    detected_text_list = []
    for file_name, file_content in resume_files.items():
        if file_name.endswith(".pdf"):
            pdf_reader = PdfReader(file_content)
            detected_text = ''
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page_obj = pdf_reader.pages[page_num]
                detected_text += page_obj.extract_text() + '\n\n'
            detected_text_list.append(detected_text)
        elif file_name.endswith(".doc") or file_name.endswith(".docx"):
            doc = Document(file_content)
            detected_text = ''
            for paragraph in doc.paragraphs:
                detected_text += paragraph.text + '\n\n'
            detected_text_list.append(detected_text) 
                
    candidates_score = []
    for text_element in detected_text_list:
        system_message_compare = f" I have a job ads with some scoring for each requirement: {descrip_score}\
        I have a cv from a candidate applying for the position.\
                Kindly provide the candidate's name and scores for each category in a json format.\
                    The scores should be the highest in each category.\
                        Please make sure the final output is json with column names,candidate's names and the categories. "
        llm_score = llm(text_element, system_message_compare, delimiter="####", print_response=True, retries=3, sleep_time=10)
        candidates_score.append(llm_score)
        
    candidates_score_text = '\n'.join(candidates_score)

    system_message_json = f" I have a json file.\
        Enclose all the json oject in [ ].\
            Note, shorten the column names but meaningful."
    llm_json = llm(candidates_score_text, system_message_json, delimiter="####", print_response=True, retries=3, sleep_time=10)
        
        
    data = json.loads(llm_json)  
    df = pd.DataFrame(data)
    df["Score"] = df.iloc[:, 1:].sum(axis=1)
    df = df.sort_values(by="Score", ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    # Display results
    num_rows = len(df)
    num_high_scores = len(df[df['Score'] > 85])
    mean_score = round(df["Score"].mean())
    highest_score = df["Score"].max()
    lowest_score = df["Score"].min()
    
    #st.header("Applicant's CV Screening 📜")

    st.write("Top 10 Job Match Candidates", df.head(10))
    st.markdown("---")
    st.write("##### Key Statistics:")
    a2, a3, a4, a5 = st.columns(4)
    #a1.image("stat.png", width=130)
    a2.metric("Total Candidates", " " ,f"{num_rows}")
    a3.metric("Mean Score", " " ,f"{mean_score}")
    a4.metric("Highest score", " ", f"{highest_score}")
    a5.metric("Lowest score", " ", f"{lowest_score}")
    style_metric_cards(border_left_color="#979249")

# Sidebar widgets for file uploads
st.sidebar.write("### Upload Job Specification")
job_spec_files = st.sidebar.file_uploader("Upload Job Specification and Scoring System", accept_multiple_files=True)

st.sidebar.write("### Upload Candidate Resumes")
resume_files = st.sidebar.file_uploader("Upload Candidate Resumes", accept_multiple_files=True)

# Process button
if st.sidebar.button("Process"):
    with st.spinner("Processing..."):
        if job_spec_files and resume_files:
            resume_contents = {file.name: file for file in resume_files}
            process_files(job_spec_files, resume_contents)
        else:
            st.error("Please upload both job specification and candidate resumes before processing.")

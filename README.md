# LLM-Powered Recruitment Tools

This repository contains two powerful Streamlit applications that utilize Large Language Models (LLMs) to assist in the recruitment process: **Rant** and **Chat**.

## Table of Contents

- [Rant](#rant)
- [Chat](#chat)
- [Installation](#installation)
- [Usage](#usage)

## Rank

**Rakt** is a tool designed to rank candidates' CVs based on a job description and a scoring system provided by the recruiter or hiring manager. This application helps streamline the process of evaluating multiple candidates by automating the ranking based on specific criteria.

### Features:

- **Upload CVs**: Upload all the candidates' CVs to the app.
- **Job Description Input**: Input the job description for which the candidates are being evaluated.
- **Scoring System**: Upload or specify the scoring system that the recruiter or hiring manager will use to evaluate the candidates.
- **Automated Ranking**: The app ranks the candidates based on the job description and the scoring system using the power of LLMs.

## Chat

**Chat** is an interactive chatbot designed to answer questions about candidates based on their CVs and the job description. It helps recruiters quickly assess which candidates possess the most relevant skills for the position.

### Features:

- **Upload CVs**: Upload the candidates' CVs for analysis.
- **Job Description Input**: Input the job description to guide the analysis.
- **Interactive Q&A**: Ask questions about the candidates, such as identifying those with the most needed skills or comparing qualifications.
- **Skill Matching**: The chatbot uses LLMs to match the skills listed in the CVs with those required by the job description, providing quick insights to the recruiter.

## Installation

To run these apps locally, you need to install the required dependencies. You can do this by using the `requirements.txt` file provided.

```bash
pip install -r requirements.txt
```
# Usage
Running the Rank App
To start the Rank app:
```bash
streamlit run rank/app.py
```
Running the chat App
To start the chat app:
```bash
streamlit run chat/app.py
```
# Usage
Uploading Files
For both apps, you will need to:

- **Upload the candidates' CVs: Ensure that the CVs are in a compatible format (e.g., PDF, DOCX).**
- **Provide the job description: Input the job description to allow the LLM to tailor its analysis.**
- **Upload the scoring system (for Rant): This could be a predefined scoring rubric or a set of criteria that the LLM will use to rank the candidates.**



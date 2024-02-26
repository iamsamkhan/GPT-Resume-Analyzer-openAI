# GPT-Resume-Analyzer-openAI

By [<b>shamshad ahmed</b>](https://iamsmakhan.netlify.app)

Connect with me on social media and explore my work:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/iamsamkhan/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/iamsamkhan)
[![Medium](https://img.shields.io/badge/Medium-Follow-03a57a?style=flat-square&logo=medium)](https://medium.com/@iamsamkhan)
[![Twitter](https://img.shields.io/twitter/follow/iamsamkhan__?style=social)](https://twitter.com/iamsamkhan__)
[![Shamshad ahmed](https://img.shields.io/badge/Sponsor-sam_khan-28a745?style=flat-square&logo=github-sponsors)](https://github.com/sponsors/iamsamkhan)

**Special Thanks to GitHub Sponsors**
ChatGPT, a powerful AI tool that can help you tailor your resume to each job application and optimize it for Applicant Tracking Systems (ATS).


## Introduction
AI Resume Screening is a tool that uses artificial intelligence to automate the process of resume screening and shortlisting. The tool uses natural language processing and machine learning algorithms to analyze resumes and classify them to the job roles based on the words in their resume.

Our tool will be designed to make the hiring process easier for both HR teams and job seekers. Our idea is to develop a user-friendly web page that enables HR teams to select the specific job role they are recruiting for. Once the job role is chosen, a link is shared with potential candidates who can then upload their resumes.

Using pre-trained machine learning models, which are trained on thousands of resumes , our tool automatically filters out resumes that do not match the job requirements. Resumes are categorized based on specific keywords and phrases related to the job role ( for example , python developer – main words can be python , developed , project , computer science etc). If a candidate's resume matches the job description, they are immediately notified and their resume is uploaded to the company's database. Even if a candidate’s resume doesn’t match the description , it shows the potential role which is suitable for the uploaded resume.

Our tool saves HR teams valuable time and energy as they no longer need to manually sift through hundreds of resumes. Job seekers also benefit from receiving immediate feedback on their eligibility, streamlining their job search process.


## Installation Steps
# Option 1: Installation from GitHub
Follow these steps to install and set up the project directly from the GitHub repository:

## Clone the Repository

Open your terminal or command prompt.
Navigate to the directory where you want to install the project.
Run the following command to clone the GitHub repository:
git clone https://github.com/iamsamkhan/GPT-Resume-Analyzer-openAI.git



## Features
1. Automated resume screening: AI Resume Screening saves time and effort by automatically screening resumes based on job requirements and pre-defined criteria.

2. Improved accuracy: The tool uses advanced algorithms to analyze resumes, reducing the likelihood of human bias and improving the accuracy of the shortlisting process.

3. Efficient process: Each resume is categorized with the model in less than 5 seconds.
   
4. Detailed output : The HR gets a detailed output of the Name, Email, Location, and the candidate's resumé, based on the scores.
   

## Usage
The code consists of the following parts:
1. app.py : This is the code which integrates the UI with the Trained model. Use the command “streamlit run app.py” which opens the web browser where the candidate can upload the resume. Alternatively the code can be hosted online using streamlit and the link can directly be sent to the candidate,to upload the resume.

2. gpt.py : It is the  which we used to train the final model on all algorithms. It also contains the accuracies of each algorithm we have used.used to extract the features from the pre-processed resume.


3. The page where HR enters the job role which is open for hiring, based on which shortlisting of candidates is done.  Use the command “streamlit run app.py” to run it in the local server.

4. It displays the resumes of the shortlisted candidates by sorting them in descending order of the scores.



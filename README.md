# Resume Analyzer

Resume Analyzer is a Streamlit application that analyzes resumes against job descriptions to provide matching scores and detailed feedback for recruiters.

## Features

- Upload resume and job description PDFs
- Generate matching scores based on skills, experience, education, and more
- View detailed breakdown of matching and missing skills
- Get recommendations for candidate evaluation

## Deployment on Streamlit Cloud

### Prerequisites

- GitHub account
- OpenAI API key

### Steps to Deploy

1. **Fork or Push the Repository to GitHub**

   Create a new GitHub repository and push this code to it.

2. **Sign up for Streamlit Cloud**

   Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign up using your GitHub account.

3. **Deploy the App**

   - Click "New app" in the Streamlit dashboard
   - Select your GitHub repository
   - Set the main file path to `streamlit_app.py`
   - Click "Deploy"

4. **Add Your OpenAI API Key (Option 1: Via Streamlit Secrets)**

   - In the Streamlit Cloud dashboard, go to your app settings
   - Under "Secrets", add your OpenAI API key in the following format:
   ```
   [openai]
   api_key = "your-openai-api-key"
   ```

5. **Using the App**

   - When using the deployed app, you'll need to enter your OpenAI API key in the sidebar
   - Upload a resume and job description PDF
   - Click "Analyze Match" to get results

## Running Locally

To run the app locally:

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

3. Open your browser and go to `http://localhost:8501`

## Notes

- The application requires an OpenAI API key to function, as it uses GPT-4o for analysis
- PDF files are processed entirely in memory and are not stored
- The application is designed to be simple to deploy and use 
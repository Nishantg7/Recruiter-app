# Resume-JD Analyzer

An AI-powered application that analyzes resumes against job descriptions to calculate a match score and provide detailed feedback for HR professionals.

## Demo

Frontend Application: [https://recruiter-app-knhd6nzm6xzff3thjx9fdk.streamlit.app/](https://recruiter-app-knhd6nzm6xzff3thjx9fdk.streamlit.app/)

## Features

- Upload resume and job description PDFs
- Get detailed match analysis
- View overall compatibility score and category-wise scores
- Receive actionable recommendations
- Easy-to-use interface

## Project Structure

- `streamlit_app.py`: Streamlit frontend application
- `app.py`: Flask backend API
- `main.py`: Core analysis logic
- Supporting modules: 
  - `pdf_processor.py`: PDF text extraction
  - `text_splitter.py`: Text chunking
  - `document_search.py`: Document search functionality
  - `conversation_chain.py`: LLM conversation handling
  - `pinecone_storage.py`: Vector database integration

## Deployment Instructions

This application has two components that need to be deployed separately:

### 1. Backend API Deployment (Render.com)

1. Create a free account on [Render.com](https://render.com/)
2. Click on "New Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - Name: recruiter-app-backend
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY_RESUME`: Your Pinecone API key for resume index
   - `PINECONE_API_KEY_JD`: Your Pinecone API key for job description index
6. Click "Create Web Service"

Render will automatically deploy your backend and provide a URL (e.g., `https://recruiter-app-backend.onrender.com`).

### 2. Frontend Deployment (Streamlit Cloud)

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Log in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `streamlit_app.py`
6. Add secrets:
   - `API_URL`: Your backend API URL from Render (e.g., `https://recruiter-app-backend.onrender.com`)
   - (Optional) `OPENAI_API_KEY`: Your OpenAI API key for local analysis
7. Click "Deploy"

Streamlit will build and deploy your app, providing a shareable URL.

## Local Development

To run the application locally:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY_RESUME=your_pinecone_api_key_resume
   PINECONE_API_KEY_JD=your_pinecone_api_key_jd
   ```
4. Run the Flask backend: `python app.py`
5. In a separate terminal, run the Streamlit frontend: `streamlit run streamlit_app.py`

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Flask
- **Text Processing**: PyPDF2
- **AI/ML**: OpenAI GPT-4o
- **Vector Database**: Pinecone
- **Deployment**: Render.com, Streamlit Cloud

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT-4o
- `PINECONE_API_KEY_RESUME`: Pinecone API key for resume index
- `PINECONE_API_KEY_JD`: Pinecone API key for job description index
- `API_URL`: Backend API URL (for Streamlit app)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
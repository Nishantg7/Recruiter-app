from flask import Flask, request, jsonify
from main import calculate_matching_score, format_analysis_output
from pdf_processor import extract_text_from_pdf
from text_splitter import split_text
from dotenv import load_dotenv
import os
from pdf_processor import extract_text_from_pdf
from text_splitter import split_text
from document_search import generate_embeddings
from conversation_chain import ConversationChain
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings  # Correct embedding model
from langchain_openai import ChatOpenAI  # Updated import path
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import json
from flask_cors import CORS
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Resume Analyzer API is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            return jsonify({"error": "OpenAI API key not found in environment variables"}), 500

        # Get files from request
        if 'resume' not in request.files or 'job_description' not in request.files:
            return jsonify({"error": "Both resume and job description files are required"}), 400
            
        resume_file = request.files['resume']
        jd_file = request.files['job_description']
        
        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as resume_temp:
            resume_file.save(resume_temp.name)
            resume_path = resume_temp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as jd_temp:
            jd_file.save(jd_temp.name)
            jd_path = jd_temp.name

        # Extract and process text
        text_resume = extract_text_from_pdf(resume_path)
        text_jd = extract_text_from_pdf(jd_path)
        
        # Clean up temporary files
        os.unlink(resume_path)
        os.unlink(jd_path)

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model="gpt-4o"
        )

        # Perform analysis
        analysis_result = calculate_matching_score(text_resume, text_jd, llm)
        json_result, score = format_analysis_output(analysis_result)
        
        try:
            # Parse the JSON string back to a dictionary
            formatted_result = {
                "analysis_json": json.loads(json_result),
                "matching_score": score
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw result
            formatted_result = {
                "analysis_json": json_result,
                "matching_score": score
            }

        return jsonify({"analysis": formatted_result})

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# For local development
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
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
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Get data from request
        resume_path = request.json.get('resume_path')
        jd_path = request.json.get('jd_path')

        # Extract and process text
        text_resume = extract_text_from_pdf(resume_path)
        text_jd = extract_text_from_pdf(jd_path)

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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
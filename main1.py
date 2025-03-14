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

def calculate_matching_score(resume_text, jd_text, llm):
    analysis_prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="""
        You are an advanced AI model designed to analyze the compatibility between a CV and a job description.
        
        CV:
        {resume}
        
        Job Description:
        {job_description}
        
        Analyze the CV against the job description and provide output in the following JSON format:
        {{
            "candidate_name": "Extracted name of the candidate from the CV",
            "contact_information": "Extracted email and phone number",
            "matching_skills": "List of skills matching the job description",
            "missing_skills": "List of important skills missing from the CV",
            "work_experience": "Total years of experience and key relevant responsibilities",
            "education": "Highest degree attained and key certifications",
            "score": "Numerical compatibility score (0-100) based on a strict comparison of qualifications, skills, and experience with the job description",
            "recommendation": {{
                "pros": "Reasons why the candidate is a good fit for the role",
                "cons": "Reasons why the candidate may not be the best fit for the role",
                "final_suggestion": "A final recommendation for HR on whether to shortlist the candidate, along with reasoning"
            }}
        }}
        
        Be thorough in your analysis and strict in your scoring. This is for professional hiring purposes.
        """
    )
    
    analysis_input = {
        "resume": resume_text,
        "job_description": jd_text
    }
    
    return llm.invoke(analysis_prompt.format(**analysis_input))

def format_analysis_output(analysis_result):
    """Format the analysis result for JSON output and extract the score"""
    try:
        content = analysis_result.content
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        json_str = content[start_idx:end_idx]
        result = json.loads(json_str)
        
        # Extract the score
        score = result.get("score", "N/A")
        
        # Return the JSON string and the score
        return json.dumps(result, indent=4), score
    except Exception as e:
        return f"Error formatting output: {str(e)}\nRaw output: {analysis_result}", "N/A"

def main():
    load_dotenv()

    api_key_resume = os.getenv("PINECONE_API_KEY_RESUME")
    api_key_jd = os.getenv("PINECONE_API_KEY_JD")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key_resume or not api_key_jd:
        raise ValueError("One or both Pinecone API keys not found in environment variables")

    # Initialize Pinecone clients
    pc_resume = Pinecone(api_key=api_key_resume, environment="gcp-starter")
    pc_jd = Pinecone(api_key=api_key_jd, environment="gcp-starter")
    
    # Initialize Ollama embeddings for embedding generation
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Initialize GPT-4 model for analysis
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model="gpt-4o"  # Ensure the correct OpenAI model is used
    )

    try:
        text_resume = extract_text_from_pdf("D:/CODING/Project/resume/ac_cv.pdf")
        text_chunks_resume = split_text(text_resume)

        text_jd = extract_text_from_pdf("D:/CODING/Project/resume/AIML_JD.pdf")
        text_chunks_jd = split_text(text_jd)

        resume_index = pc_resume.Index("resume-index")
        vectorstore_resume = PineconeVectorStore(
            index=resume_index,
            embedding=embedding_model,
            text_key="text"
        )
        
        vectorstore_resume.add_texts(
            texts=text_chunks_resume,
            metadatas=[{"source": "resume"} for _ in text_chunks_resume]
        )
        
        jd_index = pc_jd.Index("jd-index")
        vectorstore_jd = PineconeVectorStore(
            index=jd_index,
            embedding=embedding_model,
            text_key="text"
        )
        
        vectorstore_jd.add_texts(
            texts=text_chunks_jd,
            metadatas=[{"source": "job_description"} for _ in text_chunks_jd]
        )

        retriever_resume = vectorstore_resume.as_retriever(search_kwargs={'k': 2})
        retriever_jd = vectorstore_jd.as_retriever(search_kwargs={'k': 2})

        qa_resume = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_resume,
            verbose=False
        )
        
        qa_jd = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_jd,
            verbose=False
        )

        analysis_result = calculate_matching_score(text_resume, text_jd, llm)

        candidate_summary, score_str = format_analysis_output(analysis_result)

        # Convert score to integer
        try:
            score = int(score_str)
        except ValueError:
            score = None

        return candidate_summary, score

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    candidate_summary, score = main()
    print(candidate_summary)
    print(score)

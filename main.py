from pdf_processor import extract_text_from_pdf
from text_splitter import split_text
from document_search import generate_embeddings
from conversation_chain import ConversationChain
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import json

def calculate_matching_score(resume_text, jd_text, llm):
    analysis_prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="""
        You are an advanced AI model designed to analyze the compatibility between a CV and a job description and provide suggestions to assist human HR professionals in making shortlisting decisions.

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
            "work_experience": "Total years of relevant experience and key responsibilities that align with the job description",
            "education": "Highest degree attained and key certifications",
            "soft_skills": "List of relevant soft skills extracted or inferred from the CV (e.g., communication, leadership, teamwork)",
            "training_experience": "Information on any training, mentoring, or knowledge-sharing roles the candidate has undertaken",
            "adaptability": "Assessment of the candidate's potential to quickly learn missing skills based on certifications, past experiences, and technical background",
            "scoring_details": {{
                "technical_skills": "Score out of 40, based on the percentage of required technical skills present. For skills that are missing but are easy to learn, apply a reduced penalty.",
                "work_experience": "Score out of 25, reflecting the relevance and depth of the candidate's experience.",
                "education_certifications": "Score out of 15, based on academic qualifications and industry certifications.",
                "soft_skills_training": "Score out of 10, based on evidence of communication, teamwork, and training or mentoring abilities.",
                "adaptability": "Score out of 10, based on the candidate's potential to acquire missing skills efficiently."
            }},
            "score": "Total numerical compatibility score (0-100), computed as the sum of the above categories, with adjustments to reflect that some missing skills are considered trainable. Apply a penalty only if a MUST-HAVE skill is missing and not easily learnable.",
            "recommendation": {{
                "pros": "Highlight reasons why the candidate is a strong match for the role, including technical strengths, relevant experience, certifications, and soft skills.",
                "cons": "Detail any gaps or concerns, noting if any missing skills are critical versus those that can be quickly learned.",
                "final_suggestion": "Provide a balanced recommendation for HR on whether to shortlist the candidate, along with suggestions for potential areas of on-the-job training or development. Emphasize that the final decision is advisory and meant to assist in the human evaluation process."
            }}
        }}

        Additional Instructions:

        1. Technical Skills Evaluation (40%):
           - Identify all required technical skills from the job description.
           - Award proportionate points based on the number of skills present. For any missing MUST-HAVE skill, assess if it is easy to learn (e.g., Python scripting if the candidate demonstrates strong Bash experience). If easily trainable, apply a reduced penalty; otherwise, apply a standard penalty.

        2. Work Experience (25%):
           - Evaluate the total years of relevant experience and the direct applicability of key responsibilities to the job description.
           - Award higher scores when experience is directly aligned with the role's duties.

        3. Education and Certifications (15%):
           - Consider the highest degree and relevant certifications. Award full points when all critical certifications are present; otherwise, score proportionately.

        4. Soft Skills and Training Experience (10%):
           - Extract or infer soft skills such as communication, leadership, teamwork, and any training/mentoring experience.
           - If soft skills are absent, reduce the score accordingly, but note their presence in the recommendation.

        5. Adaptability (10%):
           - Assess the candidate's ability to quickly learn missing skills based on their technical background and certifications.
           - Emphasize that candidates with strong foundational expertise might overcome gaps in certain non-critical skills.

        6. Scoring Flexibility:
           - Compute the total score out of 100 using the weights provided.
           - Ensure that missing skills are penalized appropriately but allow for the possibility that some gaps are easily trainable.
           - The final score is advisory and intended to support HR professionals rather than to automatically shortlist or reject candidates.

        7. Recommendation Output:
           - Provide a balanced summary with clear pros and cons.
           - Include suggestions on how trainable gaps can be addressed during onboarding or through further development.

        Be thorough in your analysis, ensuring the evaluation mirrors a human HR professional's rigour while recognizing that some missing skills do not necessarily disqualify a candidate. Your output should assist HR in making an informed, balanced decision.
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
        model="gpt-4o"
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
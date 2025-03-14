import streamlit as st
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import io
import re

# Load environment variables
load_dotenv()

# Cache the calculation function to improve performance
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_matching_score(resume_text, jd_text, openai_api_key):
    """Calculate matching score between resume and job description"""
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model="gpt-4o"
        )
        
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
        
        analysis_result = llm.invoke(analysis_prompt.format(**analysis_input))
        return analysis_result
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def format_analysis_output(analysis_result):
    """Format the analysis result for JSON output and extract the score"""
    try:
        content = analysis_result.content
        
        # Use regex to extract the JSON part
        json_pattern = r'({.*})'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            result = json.loads(json_str)
            
            # Extract the score
            score = result.get("score", "N/A")
            
            # Return the JSON string and the score
            return {"analysis_json": result, "matching_score": score}
        else:
            return {"analysis_json": {"error": "Could not extract JSON from response"}, "matching_score": "N/A"}
    except Exception as e:
        return {"analysis_json": {"error": f"Error formatting output: {str(e)}"}, "matching_score": "N/A"}

def display_analysis_results(analysis):
    try:
        # Get the analysis data
        analysis_data = analysis.get('analysis_json', {})
        matching_score = analysis.get('matching_score', 'N/A')

        # Display matching score
        st.header("📊 Overall Match Score")
        try:
            score = float(matching_score.strip('%') if isinstance(matching_score, str) else matching_score)
            st.progress(score/100)
            st.metric("Match Score", f"{score}%")
        except (ValueError, AttributeError):
            st.warning(f"Score: {matching_score}")

        # Display candidate information
        if 'candidate_name' in analysis_data:
            st.header("👤 Candidate Information")
            st.write(f"**Name:** {analysis_data['candidate_name']}")
            if 'contact_information' in analysis_data:
                st.write(f"**Contact:** {analysis_data['contact_information']}")

        # Display skills analysis
        st.header("🎯 Skills Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matching Skills")
            if 'matching_skills' in analysis_data:
                for skill in analysis_data['matching_skills']:
                    st.success(f"✓ {skill}")
        with col2:
            st.subheader("Missing Skills")
            if 'missing_skills' in analysis_data:
                for skill in analysis_data['missing_skills']:
                    st.warning(f"⚠ {skill}")

        # Display experience and education
        if 'work_experience' in analysis_data or 'education' in analysis_data:
            st.header("📚 Experience & Education")
            if 'work_experience' in analysis_data:
                st.subheader("Work Experience")
                st.write(analysis_data['work_experience'])
            if 'education' in analysis_data:
                st.subheader("Education")
                st.write(analysis_data['education'])

        # Display soft skills and training
        if 'soft_skills' in analysis_data or 'training_experience' in analysis_data:
            st.header("🤝 Soft Skills & Training")
            if 'soft_skills' in analysis_data:
                st.subheader("Soft Skills")
                soft_skills = analysis_data['soft_skills']
                if isinstance(soft_skills, list):
                    for skill in soft_skills:
                        st.markdown(f'<div class="soft-skill">✨ {skill}</div>', unsafe_allow_html=True)
                elif isinstance(soft_skills, dict):
                    for category, skills in soft_skills.items():
                        if isinstance(skills, list):
                            for skill in skills:
                                st.markdown(f'<div class="soft-skill">✨ {skill}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="soft-skill">✨ {skills}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="soft-skill">✨ {soft_skills}</div>', unsafe_allow_html=True)
            if 'training_experience' in analysis_data:
                st.subheader("Training Experience")
                st.write(analysis_data['training_experience'])

        # Display detailed scoring
        if 'scoring_details' in analysis_data:
            st.header("📈 Detailed Scoring")
            scoring = analysis_data['scoring_details']
            cols = st.columns(5)
            
            with cols[0]:
                st.metric("Technical Skills", scoring.get('technical_skills', 'N/A'))
            with cols[1]:
                st.metric("Experience", scoring.get('work_experience', 'N/A'))
            with cols[2]:
                st.metric("Education", scoring.get('education_certifications', 'N/A'))
            with cols[3]:
                st.metric("Soft Skills", scoring.get('soft_skills_training', 'N/A'))
            with cols[4]:
                st.metric("Adaptability", scoring.get('adaptability', 'N/A'))

        # Display recommendations
        if 'recommendation' in analysis_data:
            st.header("💡 Recommendations")
            rec = analysis_data['recommendation']
            
            st.subheader("Strengths")
            if 'pros' in rec:
                for pro in rec['pros'].split('\n'):
                    if pro.strip():
                        st.success(f"✓ {pro}")

            st.subheader("Areas for Improvement")
            if 'cons' in rec:
                for con in rec['cons'].split('\n'):
                    if con.strip():
                        st.warning(f"⚠ {con}")

            if 'final_suggestion' in rec:
                st.subheader("Final Recommendation")
                st.info(rec['final_suggestion'])

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.json(analysis)  # Display raw JSON as fallback

def main():
    # Set page config
    st.set_page_config(
        page_title="Resume Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #00cc00;
        }
        .uploadLabel {
            font-size: 0.8rem;
            color: #888;
        }
        /* Improved metric styling */
        .stMetric {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 0.5rem;
            color: white !important;
        }
        .stMetric > div {
            color: white !important;
        }
        .stMetric [data-testid="stMetricLabel"] {
            color: #E0E0E0 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        .stMetric [data-testid="stMetricDelta"] {
            color: #4CAF50 !important;
        }
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        /* Header styling */
        h1, h2, h3 {
            color: #FFFFFF !important;
        }
        /* General text styling */
        p, div {
            color: #E0E0E0;
        }
        /* Success and warning colors */
        .success {
            color: #4CAF50 !important;
        }
        .warning {
            color: #FFC107 !important;
        }
        /* Soft Skills styling */
        .soft-skill {
            background-color: #2C3E50;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            margin: 0.25rem;
            display: inline-block;
            color: #E0E0E0 !important;
        }
        .soft-skill:hover {
            background-color: #34495E;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📄 Resume and Job Description Analyzer")
    st.markdown("---")

    # API Key input (hidden in sidebar)
    with st.sidebar:
        st.subheader("OpenAI API Key")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Get your API key from https://platform.openai.com/api-keys")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app analyzes resumes against job descriptions to calculate a match score and provide detailed feedback.")

    # Create two columns for file upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Job Description")
        jd_file = st.file_uploader(
            label="Upload Job Description PDF",
            type=["pdf"],
            key="jd",
            help="Upload a PDF file containing the job description"
        )

    with col2:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader(
            label="Upload Resume PDF",
            type=["pdf"],
            key="resume",
            help="Upload a PDF file containing the resume"
        )

    # Analysis button with loading state
    if st.button("🔍 Analyze Match", type="primary"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            return
        
        if not (resume_file and jd_file):
            st.error("Please upload both resume and job description files.")
            return

        with st.spinner("Analyzing documents..."):
            try:
                # Extract text from PDF files
                resume_text = extract_text_from_pdf(resume_file)
                jd_text = extract_text_from_pdf(jd_file)
                
                # Perform the analysis
                analysis_result = calculate_matching_score(resume_text, jd_text, openai_api_key)
                
                if analysis_result:
                    formatted_result = format_analysis_output(analysis_result)
                    
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")
                    display_analysis_results(formatted_result)
                else:
                    st.error("❌ Error in analysis")
                    st.error("Failed to get analysis result")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Made with ❤️ by Your Resume Analyzer</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
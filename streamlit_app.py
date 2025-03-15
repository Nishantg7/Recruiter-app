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
import requests
import time

# Add a config file
st.set_page_config(
    page_title="Resume-JD Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2C3E50;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #3498DB;
    }
    .metric-label {
        color: #E0E0E0;
        font-size: 14px;
    }
    .metric-value {
        color: white;
        font-size: 20px;
        font-weight: bold;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .soft-skill {
        display: inline-block;
        background-color: #2C3E50;
        color: #E0E0E0;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 0.9em;
        transition: background-color 0.3s;
    }
    .soft-skill:hover {
        background-color: #34495E;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Define API URLs - this will be the URL of your deployed backend
API_URL = st.secrets.get("API_URL", "https://recruiter-app-backend.onrender.com")

# Define the backend API URL
BACKEND_API_URL = "https://recruiter-app-backend.onrender.com"

# Cache the calculation function to improve performance
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_matching_score_api(resume_file, jd_file):
    """Calculate matching score using the deployed API"""
    try:
        # Prepare the files for the API request
        files = {
            'resume': ('resume.pdf', resume_file, 'application/pdf'),
            'job_description': ('jd.pdf', jd_file, 'application/pdf')
        }
        
        # Send the request to the API
        with st.spinner('Analyzing... This may take up to 2 minutes'):
            st.info("Making API call to backend service...")
            response = requests.post(f"{API_URL}/analyze", files=files, timeout=180)
            
            if response.status_code != 200:
                st.error(f"API Error: Status code {response.status_code}\nResponse: {response.text}")
                return None, None
                
            # Get the raw result
            raw_result = response.json()
            
            # Check the structure - this is debugging code we can keep for now
            if 'analysis' in raw_result:
                # Original expected structure
                result = raw_result
                analysis_json = result.get('analysis', {}).get('analysis_json')
                matching_score = result.get('analysis', {}).get('matching_score')
            else:
                # Direct structure from the API
                # The API seems to be returning the analysis data directly at the top level
                st.info("Using direct API response structure")
                analysis_json = raw_result  # The whole response is the analysis JSON
                matching_score = raw_result.get('score')  # Score is directly in the response
                
            # Convert the data structure to what the display function expects
            formatted_result = {
                'analysis_json': analysis_json if analysis_json else raw_result,
                'matching_score': matching_score if matching_score else raw_result.get('score')
            }
            
            return formatted_result, formatted_result.get('matching_score')
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend service might be overloaded or starting up.")
        return None, None
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the backend API. The service might be down or unreachable.")
        return None, None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        return None, None

# Original local calculation function - kept as fallback
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
        st.info("Processing analysis results...")
        
        # Check if analysis is None
        if analysis is None:
            st.error("No analysis results to display. The analysis process failed.")
            return
            
        # Get the analysis data
        if isinstance(analysis, dict):
            # It's already a dict, get direct or nested data
            if 'analysis_json' in analysis:
                # Using the nested structure
                analysis_data = analysis.get('analysis_json', {})
                matching_score = analysis.get('matching_score', 'N/A')
            else:
                # Direct API response
                analysis_data = analysis
                matching_score = analysis.get('score', 'N/A')
        else:
            # Try to parse if it's a string
            try:
                analysis_data = json.loads(analysis)
                matching_score = 'N/A'
            except json.JSONDecodeError:
                st.error(f"Failed to parse analysis string")
                return
        
        # If analysis_data is still a string, try to parse it
        if isinstance(analysis_data, str):
            try:
                analysis_data = json.loads(analysis_data)
            except json.JSONDecodeError:
                st.error("Failed to parse nested JSON string")
                return

        # Display matching score
        st.header("📊 Overall Match Score")
        try:
            # Try multiple possible score fields
            score_value = None
            
            # Look for score in different locations
            if matching_score != 'N/A':
                score_value = matching_score
            elif 'score' in analysis_data:
                score_value = analysis_data['score']
            elif 'matching_score' in analysis_data:
                score_value = analysis_data['matching_score']
                
            if score_value:
                # Convert to float and remove % if present
                if isinstance(score_value, str) and '%' in score_value:
                    score = float(score_value.strip('%'))
                else:
                    score = float(score_value)
                
                st.progress(score/100)
                st.metric("Match Score", f"{score}%")
            else:
                st.warning(f"Score: N/A")
        except (ValueError, AttributeError, TypeError) as e:
            st.warning(f"Could not process score: {str(e)}")

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
                matching_skills = analysis_data['matching_skills']
                if isinstance(matching_skills, list):
                    for skill in matching_skills:
                        st.success(f"✓ {skill}")
                elif isinstance(matching_skills, str):
                    for skill in matching_skills.split(','):
                        if skill.strip():
                            st.success(f"✓ {skill.strip()}")
        
        with col2:
            st.subheader("Missing Skills")
            if 'missing_skills' in analysis_data:
                missing_skills = analysis_data['missing_skills']
                if isinstance(missing_skills, list):
                    for skill in missing_skills:
                        st.warning(f"⚠ {skill}")
                elif isinstance(missing_skills, str):
                    for skill in missing_skills.split(','):
                        if skill.strip():
                            st.warning(f"⚠ {skill.strip()}")

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
        st.error(f"Error displaying analysis results: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        # Also display raw data for debugging
        st.subheader("Raw Analysis Data")
        st.json(analysis)

def check_backend_health():
    """Check if the backend API service is up and running"""
    try:
        st.info(f"Checking backend health at {BACKEND_API_URL}...")
        response = requests.get(f"{BACKEND_API_URL}/", timeout=10)
        
        if response.status_code == 200:
            st.success("Backend API is online and responding.")
            return True
        else:
            st.error(f"Backend API returned status code {response.status_code}")
            st.error(f"Response: {response.text[:200]}...")
            return False
    except requests.exceptions.Timeout:
        st.error("Backend API check timed out. The service might be starting up or overloaded.")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the backend API. The service might be down or unreachable.")
        return False
    except Exception as e:
        st.error(f"Error checking backend health: {str(e)}")
        return False

def analyze_resume_with_api(resume_file, jd_file):
    try:
        # Prepare the files for the API request
        files = {
            'resume': resume_file,
            'job_description': jd_file
        }
        
        # Send POST request to the API
        response = requests.post(f"{BACKEND_API_URL}/analyze", files=files)
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend service: {str(e)}")
        return None

def main():
    st.title("Resume - JD Analyzer")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📊 Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Resume and Job Description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
            
        with col2:
            jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
        
        api_mode = st.radio(
            "Analysis Mode",
            ["Use Remote API (Recommended)", "Use Local Analysis"],
            index=0,
            help="Remote API is more reliable but requires internet connection. Local analysis uses your OpenAI API key."
        )
        
        if api_mode == "Use Local Analysis":
            openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        else:
            openai_api_key = None
            
        analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button:
            if not resume_file or not jd_file:
                st.error("Please upload both resume and job description files.")
            elif api_mode == "Use Local Analysis" and not openai_api_key:
                st.error("Please enter your OpenAI API key for local analysis.")
            else:
                with st.spinner("Analyzing... This may take up to 2 minutes"):
                    # Check if backend API is available for remote analysis
                    if api_mode == "Use Remote API (Recommended)":
                        backend_available = check_backend_health()
                        if not backend_available:
                            st.error("Backend API is not available. Please try using Local Analysis or try again later.")
                            return
                        
                        st.info("Backend API is available. Proceeding with analysis...")
                        
                        # Reset the file pointers to the beginning
                        resume_file.seek(0)
                        jd_file.seek(0)
                        
                        # Call API and get results
                        analysis, score = calculate_matching_score_api(resume_file, jd_file)
                        
                        # Raw output for debugging
                        st.subheader("Debug: API Response")
                        with st.expander("View Raw Response Data"):
                            st.json(analysis)
                        
                        # Check if analysis was successful
                        if analysis is None:
                            st.error("Analysis failed. Please check the error messages above or try Local Analysis.")
                            return
                            
                        # Log what we received for debugging
                        st.info(f"Received analysis type: {type(analysis)}")
                        if isinstance(analysis, dict):
                            st.info(f"Analysis keys: {list(analysis.keys())}")
                            
                    else:  # Local Analysis
                        # Extract text from PDF files
                        resume_text = extract_text_from_pdf(resume_file)
                        jd_text = extract_text_from_pdf(jd_file)
                        
                        # Perform analysis
                        analysis_result = calculate_matching_score(resume_text, jd_text, openai_api_key)
                        analysis_json, score = format_analysis_output(analysis_result)
                        
                        try:
                            analysis = json.loads(analysis_json)
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing analysis result: {e}")
                            st.error(f"Raw result: {analysis_json[:200]}...")  # Show first 200 chars
                            analysis = None
                    
                    # Display analysis results if successful
                    if analysis:
                        try:
                            display_analysis_results(analysis)
                        except Exception as e:
                            st.error(f"Error in display_analysis_results: {str(e)}")
                            st.error("Raw analysis data:")
                            st.json(analysis)
                    else:
                        st.error("No analysis data was generated. Please try again or switch analysis modes.")
        
        # Show example at bottom of page
        with st.expander("About this tool"):
            st.markdown("""
            This tool uses AI to analyze how well a resume matches a job description. It provides a matching score
            and detailed feedback to help HR professionals make better shortlisting decisions.
            
            **How it works:**
            1. Upload a resume and job description (PDF format)
            2. Choose your analysis mode (Remote API or Local)
            3. Click Analyze
            4. View the detailed results and recommendations
            
            **Note:** Analysis takes about 1-2 minutes to complete.
            """)
    
    with tab2:
        st.markdown("""
        # About Resume-JD Analyzer
        
        This application helps HR professionals and recruiters evaluate how well a candidate's resume matches a job description.
        
        ## Features
        
        - **Detailed Analysis:** Evaluates technical skills, work experience, education, soft skills, and adaptability
        - **Scoring System:** Provides a numerical score for each category and an overall match percentage
        - **Recommendations:** Highlights strengths and areas for improvement
        - **Easy to Use:** Simple interface for uploading documents and viewing results
        
        ## How It Works
        
        The application uses advanced AI to:
        1. Extract text from resume and job description PDFs
        2. Analyze key components and requirements
        3. Calculate a matching score based on various factors
        4. Provide detailed feedback and recommendations
        
        ## Privacy
        
        - Documents are processed securely
        - No data is stored permanently
        - Analysis is performed on-demand only
        
        ## Feedback
        
        We welcome your feedback to improve this tool! Please reach out with any suggestions or issues.
        """)

if __name__ == "__main__":
    main()
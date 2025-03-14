from pdf_processor import extract_text_from_pdf
from text_splitter import split_text
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json
import os
from dotenv import load_dotenv


def extract_and_label_resumes(resume_folder, llm, output_file, api_keys, current_key_index):
    """
    Extract text from resumes, label them with the LLM, and save to a JSONL file for fine-tuning.
    """
    labeled_data = []
    resumes = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder) if f.endswith('.pdf')]

    for resume_file in resumes:
        while True:
            try:
                # Extract text from PDF
                resume_text = extract_text_from_pdf(resume_file)
                
                # Process with the LLM for ATS-relevant fields
                labeled_output = extract_ats_fields(resume_text, llm)
                
                if labeled_output:
                    # Prepare JSONL entry
                    labeled_entry = {
                        "prompt": f"Extract key details from this resume:\n\n{resume_text}",
                        "completion": labeled_output
                    }
                    labeled_data.append(labeled_entry)
                break  # Exit the loop if successful
            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message:
                    print(f"Rate limit reached for API key {current_key_index + 1}. Switching keys...")
                    llm, current_key_index = switch_api_key(api_keys, current_key_index)
                else:
                    print(f"Error processing {resume_file}: {str(e)}")
                    break  # Exit the loop on non-rate limit errors
    
    # Save labeled data to JSONL
    with open(output_file, 'w') as f:
        for entry in labeled_data:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Labeled data saved to {output_file}")


def extract_ats_fields(resume_text, llm):
    """
    Extract ATS-relevant fields using the LLM.
    """
    analysis_prompt = PromptTemplate(
        input_variables=["resume"],
        template="""
        You are an advanced AI model designed to extract key details from resumes for ATS (Applicant Tracking System) purposes. Extract the following fields from the provided resume and output the result in JSONL format, with each entry being a single JSON object. Be strict about the data extracted, and fill any missing fields with 'N/A'. Ensure all values are token-efficient, concise, and relevant. Only include necessary information, without additional commentary or explanation. Each resume should have one line in the output.

        Resume:
        {resume}

        Extract the following fields in JSON format:
        {{
            "Name": "Full name of the candidate (First and Last Name, formatted correctly)",
            "Email": "Primary email address (standard format: name@example.com, if present)",
            "Phone": "Primary phone number (numeric format, including area code, omit non-numeric characters)",
            "Experience": "Total years of professional experience and key industries/domains (e.g., '5 years in Software Development, specializing in AI and Machine Learning')",
            "Education": "Highest degrees attained, universities attended, and graduation years (e.g., 'BSc in Computer Science, University of XYZ, 2019')",
            "Skills": "Key technical and soft skills (e.g., 'Python, Java, Leadership, Project Management')"
        }}
        """
    )
    
    try:
        analysis_input = {"resume": resume_text}
        analysis_result = llm.invoke(analysis_prompt.format(**analysis_input))
        
        # Extract JSON output from the response
        content = analysis_result.content
        print("LLM Response Content:", content)  # Debugging line

        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        json_str = content[start_idx:end_idx]
        print("Extracted JSON String:", json_str)  # Debugging line

        labeled_output = json.loads(json_str)
        
        return json.dumps(labeled_output)
    except Exception as e:
        print(f"Error extracting fields: {str(e)}")
        return None


def initialize_llm(api_keys):
    current_key_index = 0
    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_keys[current_key_index],
        model_name="llama-3.1-70b-versatile"
    )
    return llm, current_key_index


def switch_api_key(api_keys, current_key_index):
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"Switching to API key {current_key_index + 1}")
    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_keys[current_key_index],
        model_name="llama-3.1-70b-versatile"
    )
    return llm, current_key_index


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load API keys
    api_keys = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2")
    ]
    if not all(api_keys):
        raise ValueError("One or more GROQ_API_KEYs not found in environment variables")

    # Initialize LLM
    llm, current_key_index = initialize_llm(api_keys)

    # Folder containing resumes (update this path as needed)
    resume_folder = "D:/CODING/Project/dataset/output"
    
    # Output file for labeled data
    output_file = "labeled_resumes.jsonl"

    # Extract and label resumes
    extract_and_label_resumes(resume_folder, llm, output_file, api_keys, current_key_index)


if __name__ == "__main__":
    main()

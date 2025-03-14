from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone.exceptions import PineconeApiException

def create_pinecone_indices(api_key_resume, api_key_jd):
    pc_resume = Pinecone(api_key=api_key_resume)
    pc_jd = Pinecone(api_key=api_key_jd)
    
    # Attempt to create the resume index
    try:
        pc_resume.create_index(
            name="resume-index",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Created resume index.")
    except PineconeApiException as e:
        if e.code == "ALREADY_EXISTS":
            print("Resume index already exists. Continuing with existing index.")
        else:
            raise  # Re-raise the exception if it's not the expected one

    # Attempt to create the job description index
    try:
        pc_jd.create_index(
            name="jd-index",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Created job description index.")
    except PineconeApiException as e:
        if e.code == "ALREADY_EXISTS":
            print("Job description index already exists. Continuing with existing index.")
        else:
            raise  # Re-raise the exception if it's not the expected one

def upsert_data_to_index(pc, index_name, data):
    index = pc.Index(index_name)
    index.upsert(vectors=data)
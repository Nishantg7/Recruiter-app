from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

GROQ_KEY = os.getenv("GROQ_API_KEY")

class ConversationChain:
    def __init__(self, retriever):
        if retriever is None:
            raise ValueError("retriever cannot be None")
        self.retriever = retriever

    def create_chain(self):
        # Define the prompt template for the language model
        prompt_template = """
            Context: {context}
            Question: {question}
            Answer the question based on the provided context. If the context doesn't contain enough information, say "I don't know."
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Initialize the language model with specific parameters
        llm = ChatGroq(temperature=0, groq_api_key=GROQ_KEY, model_name="llama-3.1-70b-versatile")

        # Create a RetrievalQA chain using the language model and document retriever
        return RetrievalQA.from_llm(
            llm=llm, 
            retriever=self.retriever, 
            prompt=PROMPT
        ) 

import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Gemini configuration
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Initialize Gemini
client = genai.Client(api_key=API_KEY)


# STEP 1: Create RAG prompt
def create_rag_prompt(question, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an Enterprise Document Question Answering Assistant.

Instructions:
1. Answer using the provided document context.
2. Understand abbreviations and equivalent terms.
3. Do not invent facts.
4. If information is missing, clearly say so.
5. Give accurate and relevant answers.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""

    return prompt


# STEP 2: Generate answer using Gemini
def generate_answer(question, contexts):

    if not contexts:
        return "No relevant information retrieved."

    prompt = create_rag_prompt(question, contexts)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return "Gemini could not generate an answer."

    except Exception as e:
        print("Gemini Error:", e)
        return "Error generating answer."


# STEP 3: Test Gemini integration
if __name__ == "__main__":

    sample_question = "What is Retrieval-Augmented Generation?"

    sample_contexts = [
        "RAG stands for Retrieval-Augmented Generation.",
        "RAG combines information retrieval with large language models.",
        "It retrieves relevant documents and uses them as context "
        "to generate accurate answers."
    ]

    print("\nTesting Gemini RAG Integration...")

    answer = generate_answer(
        sample_question,
        sample_contexts
    )

    print("\nQUESTION:")
    print(sample_question)

    print("\nANSWER:")
    print(answer)

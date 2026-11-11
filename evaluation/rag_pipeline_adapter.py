
from src.retriever import Retriever
from src.gemini_llm import generate_answer as gemini_generate_answer


# Initialize FAISS retriever
retriever = Retriever()


def detect_department(question: str):
    """
    Identify the department based on question keywords.
    """

    question = question.lower()

    department_keywords = {
        "hr": [
            "leave",
            "annual leave",
            "sick leave",
            "casual leave",
            "employee",
            "salary",
            "human resources",
        ],

        "finance": [
            "travel",
            "expense",
            "reimbursement",
            "finance",
            "hotel",
            "flight",
        ],

        "it": [
            "computer",
            "laptop",
            "technical",
            "software",
            "password",
            "internet",
            "it support",
        ],
    }

    for department, keywords in department_keywords.items():
        if any(keyword in question for keyword in keywords):
            return department

    return None


def retrieve_contexts(question: str) -> list[str]:

    department = detect_department(question)

    results = retriever.retrieve(
        query=question,
        top_k=3,
        department=department
    )

    contexts = []

    for result in results:

        document = result["document"]

        if isinstance(document, str):
            text = document

        elif isinstance(document, dict):
            text = (
                document.get("text")
                or document.get("content")
                or document.get("page_content")
            )

        else:
            text = (
                getattr(document, "page_content", None)
                or getattr(document, "text", None)
            )

        if text:
            contexts.append(text)

    return contexts


def generate_answer(question: str, contexts: list[str]) -> str:

    return gemini_generate_answer(question, contexts)


def run_rag_pipeline(question: str) -> dict:

    contexts = retrieve_contexts(question)

    if not contexts:
        return {
            "question": question,
            "contexts": [],
            "answer": "No relevant context was retrieved."
        }

    answer = generate_answer(question, contexts)

    return {
        "question": question,
        "contexts": contexts,
        "answer": answer
    }

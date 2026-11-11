
from src.rag_pipeline import RAGPipeline


# Initialize the complete RAG application
rag = RAGPipeline()


# Ask an employee question
question = "How many annual paid leave days do employees receive?"

result = rag.ask(
    question=question,
    top_k=3
)


print("\n" + "=" * 80)
print("TECHNOVA ENTERPRISE KNOWLEDGE ASSISTANT")
print("=" * 80)

print("\nEMPLOYEE QUESTION:")
print(result["question"])

print("\nGENERATED ANSWER:")
print(result["answer"])

print("\nRETRIEVED SOURCE DOCUMENTS:")
print("-" * 80)

if result["sources"]:

    for source in result["sources"]:

        print("\nSource:", source["source"])
        print("Department:", source["department"])
        print("Page:", source["page"])
        print("Chunk ID:", source["chunk_id"])
        print("Similarity:", source["similarity_score"])

else:
    print("No supporting documents were retrieved.")



from src.rag_pipeline import RAGPipeline


rag = RAGPipeline()


employee_questions = [
    "How many annual paid leave days do employees receive?",

    "How can I reset my company password?",

    "What are the IT helpdesk working hours?",

    "What is the daily meal allowance for business travel?",

    "When should I submit my travel reimbursement claim?",

    "What is the company's international relocation policy?"
]


for question in employee_questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    result = rag.ask(
        question=question,
        top_k=3
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSUPPORTING SOURCES:")

    if result["sources"]:

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"({source['department']})"
            )

    else:
        print("- No documents retrieved.")


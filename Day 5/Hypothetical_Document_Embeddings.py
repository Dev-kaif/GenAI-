from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import os

Gemini_key = os.getenv("GEMINI")

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=Gemini_key
)

client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=Gemini_key,
    temperature=0.7,
)


def do_similarity_serach(query):

    retriver = QdrantVectorStore.from_existing_collection(
        collection_name="learning_langchain",
        url="http://localhost:3000",
        embedding=embedder,
    )

    search = retriver.similarity_search(query)

    return search




def call_gemini_real(query,context):

    messages = [
        SystemMessage(
            content="""
                You are a **PDF Assistant** — respond as if you *are* the PDF itself.

                Your rules:
                - Base answers **only on retrieved content**.
                - Cite exact **page numbers** (e.g., [page 14]) when stating facts.
                - Suggest **pages** for further reading when needed.
                - If answer is not found in content, say: "I’m sorry, I couldn’t find that in the document."
                - Improvise only if the user’s question **relates to the PDF** (e.g., missing code snippet) — but clearly say: _"This part is generated based on context, not in the original PDF."_
                - If the question is totally **unrelated**, just say it's out of context and **roast the user in GenZ slang** (like a savage redditor).
            """
        ),
        HumanMessage(
            content=f"""
                Here is the context retrieved from the PDF: {context}
                User question: {query}
            """
        ),
    ]

    response = client.invoke(messages)
    
    print("\n\n context :", context, "\n\n")
    
    print("\n\n real :", response.content, "\n\n")
    


def call_gemini_for_gernrating_fake_response_for_symentic_search(query):

    system_prompt = [
        SystemMessage(
            content="You are a helpful AI assistant. Your task is to provide a comprehensive and detailed response to the user's question. Use the provided query to generate a summary, key details, and relevant information."
        ),
        HumanMessage(content=f"Question: {query}"),
    ]

    response = client.invoke(system_prompt)
    print("\n\n fake: ", response.content, "\n\n")
    return response.content


def __main__():
    while True:
        user = input("query >>> ")

        if "end" in user:
            print("ending the conversation")
            break
        
        fake_response = call_gemini_for_gernrating_fake_response_for_symentic_search(query=user)
        
        retrived_docs = do_similarity_serach(fake_response)
        
        call_gemini_real(query=user,context=retrived_docs)


__main__()

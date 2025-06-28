import os
import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


Gemini_key = os.getenv("GEMINI")
my_pdf = Path(__file__).parent / "how-to-code-in-go.pdf"
loader = PyPDFLoader(my_pdf)

my_docs = loader.load()

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=Gemini_key
)


client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=Gemini_key,
    temperature=0.7,
)

# Global similar array
all_retrievals = []

def call_gemini_for_parallel_query(query):

    messages = [
        SystemMessage(
            content="""
            You are a **expert Prompt writing engineer** .
            **Rules for Output Generation:** :
            - Genrate three similar queries from the user query but in very detail and very less abstract manner
            - Only genrate three queries 
            - Output should be  JSON format with key "Genrated" inside which has value of array of strings
            - ***NEVER INCLUDE ```json ``` OR ANYTHING*** IT SHOULD ALWAYS BE CLEAN
            - Do not include any ```json or ``` in your response.
            - Do not include any explanations, headings, or additional text.
            - Always output only raw JSON.
            - Always return the original query in the array of genrated query as well
            - so total count of stings in the array will be **4**

            example : 

            Input : "How to train a transformer model?"
            Output : 
            {
                "Genrated":["How to train a transformer model?","Best practices for fine-tuning transformers", "Transformer architecture training guide", "Steps to train BERT or GPT models"]
            }
        """
        ),
        HumanMessage(
            content=f"""
                User question: {query}
            """
        ),
    ]

    response = client.invoke(messages)
    cleaned = re.sub(r"```json|```", "", response.content).strip()
    res= json.loads(cleaned)
    
    multiple_queries = res["Genrated"]
    
    return multiple_queries



def chunks_maker(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(documents=docs)

    # print("docs : ",len(docs)) ==> # docs :  447
    # print("sp_docs : ",len(split_docs)) ==> # sp_docs :  772

    return split_docs


def add_vector(doc):
    vector_Store = QdrantVectorStore.from_documents(
        documents=[],
        collection_name="learning_langchain",
        url="http://localhost:3000",
        embedding=embedder,
    )

    vector_Store.add_documents(documents=doc)
    print("done")
    return

# splied = chunks_maker(docs=my_docs)

# add_vector(splied)


def format_context_for_gemini(unfiltered_similar):
    # Flatten first
    flat_docs = [doc for sublist in unfiltered_similar for doc in sublist]
    # Deduplicate
    seen = {}
    unique_docs = []
    for doc in flat_docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen[key] = True
            unique_docs.append(doc)
    # Build readable context
    context = "\n\n".join(
        f"[page {doc.metadata.get('page')}] {doc.page_content}"
        for doc in unique_docs
    )
    return context


def do_similarity_serach(query):

    retriver = QdrantVectorStore.from_existing_collection(
        collection_name="learning_langchain",
        url="http://localhost:3000",
        embedding=embedder,
    )

    search = retriver.similarity_search(query)

    return search

def do_similarity_serach_for_multiple(query):
    result = do_similarity_serach(query=query)
    return result


def call_gemini(query,context):

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
    print("\n\n", response.content, "\n\n")



def __main__():
    while True:
        user = input("query >>> ")

        if "end" in user:
            print("ending the coversation")
            break

        multiple_result =  call_gemini_for_parallel_query(user)
        
        all_retrievals.clear()  # clear before new retrieval
        for result in multiple_result:
            retrieved = do_similarity_serach_for_multiple(result)
            all_retrievals.append(retrieved)
        
        unique_context = format_context_for_gemini(unfiltered_similar=all_retrievals)
        
        call_gemini(query=user,context=unique_context)

__main__()

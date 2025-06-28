import os
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


split_docs = chunks_maker(my_docs)

add_vector(split_docs)


def do_similarity_serach(query):

    retriver = QdrantVectorStore.from_existing_collection(
        collection_name="learning_langchain",
        url="http://localhost:3000",
        embedding=embedder,
    )

    search = retriver.similarity_search(query)

    return search


def call_gemini(query):

    client = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=Gemini_key,
        temperature=0.7,
    )

    similar = do_similarity_serach(query=query)

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
                Here is the context retrieved from the PDF: {similar}
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

        call_gemini(query=user)


__main__()

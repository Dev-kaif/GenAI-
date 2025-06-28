from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

Gemini_key = os.getenv("GEMINI")


client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=Gemini_key,
    temperature=0.7,
)



def call_gemini_step_back(user_query):
    # Step 1: Summarization of the query
    step1_msgs = [
        SystemMessage(content="""
        You are an expert at world knowledge. 
        Your task is to step back and paraphrase a question to a more generic 
        step-back question, which is easier to answer. 

        Here are a few examples:
        Original Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956?
        Stepback Question: Which positions have Knox Cunning- ham held in his career?

        Original Question: Who was the spouse of Anna Karina from 1968 to 1974?
        Stepback Question: Who were the spouses of Anna Karina?

        Original Question: Which team did Thierry Audel play for from 2007 to 2008?
        Stepback Question: Which teams did Thierry Audel play for in his career?
                      """),
        HumanMessage(content=f"User Query: {user_query}")
    ]
    step1_response = client.invoke(step1_msgs)
    summarized_query = step1_response.content.strip()


    # Step 2: Reasoning steps
    step2_msgs = [
        SystemMessage(content="You are a reasoning assistant. Break down how to answer the following question step-by-step."),
        HumanMessage(content=f"Question: {summarized_query}")
    ]
    step2_response = client.invoke(step2_msgs)
    reasoning_steps = step2_response.content.strip()
    

    # Step 3: Final Answer
    step3_msgs = [
        SystemMessage(content="You are a helpful assistant. Using the following reasoning, provide a final clear answer."),
        HumanMessage(content=f"Reasoning Steps:\n{reasoning_steps}\n\nAnswer the original question: {user_query}")
    ]
    step3_response = client.invoke(step3_msgs)

    print("Summarized Query:", summarized_query)
    print("Reasoning Steps:", reasoning_steps)
    print("Final Answer:", step3_response.content)

    # return step3_response.content



def __main__():
    while True:
        user = input("query >>> ")

        if "end" in user:
            print("ending the conversation")
            break

        call_gemini_step_back(user_query=user)


__main__()
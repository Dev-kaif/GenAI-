from googlesearch import search
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import requests


load_dotenv()

Gemini_key = os.getenv("GEMINI")

client = genai.Client(api_key=Gemini_key)


def google_search_and_scrape(query, num_results=5):
    print(f"Searching Google for: {query}")
    results = []

    for url in search(query, num_results=num_results):
        try:
            # print(f"\nFetching: {url}")
            response = requests.get(
                url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
            )
            soup = BeautifulSoup(response.text, "html.parser")

            # Try to get main text content
            paragraphs = soup.find_all("p")
            page_text = "\n".join(p.get_text() for p in paragraphs)

            results.append(
                {
                    "url": url,
                    "content": page_text.strip()[
                        :1000
                    ],  # limit preview to 1000 characters
                }
            )

        except Exception as e:
            continue

    return results


def get_weather(city):
    print("__🛠️ __ is called ")
    url = f"https://wttr.in/{city}?format=%C+%t%22"
    response = requests.get(url)
    if response.status_code == 200:
        return f"the weather in {city} is {response.text}"


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name and returns the current weather of that city",
    },
    "google_search_and_scrape": {
        "fn": google_search_and_scrape,
        "description": "Takes a query parameter and seraches on internet on google serach engine and scraps the data and returns the json with key url and content",
    },
}

system_prompt = """
You are an ai assistant who is expert in resolving user query with the help of available toos
For the given user input and available tools plan the step by step execution,based on planning.
Select the relavenrt tool from the available tool. based on tool selection
Wait for the observation and based on observation from the tool call resolve the user query

Follow the steps in sequence that is "plan", "action", "observe","reselt" mode

**Rules for Output Generation:**
1.  **Strict JSON Format:** Your output MUST be a single, valid JSON object.
    * **NO extra text:** Do NOT include any conversational text, explanations, or filler outside the JSON object (e.g., "Here is your analysis:", "```json", "```").
    * **Correct Syntax:** All keys and string values MUST be enclosed in double quotes (e.g., {"key":"value"}).
    * **Proper Escaping:** Newline characters must be escaped as '\\n' inside string values. Other special characters in strings must also be properly escaped.
2.  **Sequence Adherence:** Follow the steps in sequence: "plan", "action", "observe","result".
3.  **Step-by-Step Execution:** Always perform one step at a time and wait for next input. Do NOT output multiple JSON objects in a single turn.
4.When giving the result never mention about the tools or what you did in bckgroud such as seraching or fetching 

Output JSON format: 
{{
    "step":"string",
    "content":"string",
    "function":"the name of function if the step is action",
    "input":"The input parameter for the function",
}}

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name and returns the current weather of that city",
    },
    "google_search_and_scrape": {
        "fn": google_search_and_scrape,
        "description": "Takes a query parameter and seraches on internet on google serach engine and scraps the data and returns the json with key url and content",
    },
}

Example:
Input: What is weather in New York?
Output:{{step:"plan",content:"The user is asking wheather condition of new york city"}}
Output:{{
 "step": "plan",
 "content": "From the available tools i should call get_weather"}}
Output : {{
 "step": "action",
 "function":"get_weather",
 "input":"new york"
 "content":"Featching the weather deatils....."
}}
Output:{{"step":"observe","output":"12 Degree Celcious"}}
Output:{{"step":"result","content":"the weather of new york seems to be 12 degress"}}

"""

user_prompt = []

while True:
    user = input("your query > ")
    user_prompt.append(user)

    while True:

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                temperature=0.8,
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
            contents=user_prompt,
        )
        user_prompt.append(response.text)
        parsed_response = json.loads(response.text)

        if parsed_response.get("step") != "result" and "observe":
            print("🧠 : ", parsed_response.get("content"))

        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            if available_tools.get(tool_name):
                output = available_tools[tool_name].get("fn")(tool_input)
                user_prompt.append(json.dumps({"step": "observe", "output": output}))
                continue

        if parsed_response.get("step") == "result":
            print("🤖: ", parsed_response.get("content"))
            break

        continue

    if "end" in user:
        print("Ending the Chat ❌ ")
        break
    continue

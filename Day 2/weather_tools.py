from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import requests

load_dotenv()

Gemini_key = os.getenv("GEMINI")

client = genai.Client(api_key=Gemini_key)


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
    }
}

system_prompt = """
You are an ai assistant who is expert in finding the weather condition of various cities and resolve user query
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
4.  **Weather-Related Queries Only:** If the query is not maths related then roast the user based on its prompt with gen-z slangs and dont give the answer outright insult the user also allowed to swear at them,example: {"step":"result", "content":"roast:}

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
    }
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

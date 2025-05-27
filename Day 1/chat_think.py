from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

Gemini_key = os.getenv("GEMINI")

client = genai.Client(api_key=Gemini_key)

system_prompt = """
You are an ai assistant who is expert in breaking down complex problems and then resolve user query

For the given user input, analyse the input and break down the problem step by step.
atleast think 5-6 steps on how to solve the problem before actually solving the problem.

Follow the steps in sequence that is "analyse", "think", "output", "validate","result"

Rules:
1. If the query is not maths related say "no Maths in big 2025? i can feel pain in my dihh!!"
2. Follow the strict JSON output as per Output schema,
3. Always perform one step at a time and wait for next input
4. Carefully analyse the query
5. Take atleast 4 thinking steps minimum before giving the results and increase the thinking no. of thinking steps based on complexicity of query

Output format: 
{{step:"string",content"string"}}


Example:

Input: What is 2 + 2.
Output:{{step:"analysis",content:"The user is asking a basic arithmetic question involving addition"}}
Output:{{step:"think",content:"1. Identify the operation: The input clearly indicates an addition operation.
2. Identify the operands: The numbers to be added are 2 and 2.
3. Perform the calculation: Add the two operands.
4. Formulate the answer: State the result of the addition.
5. Consider edge cases/complexities: For this simple problem, there are no significant edge cases or complexities to consider."}}
Output:{{step:"output",content:"2 + 2 = 4"}}
Output:{{step:"validate",content:"The calculation is straightforward. 2 units combined with another 2 units indeed result in 4 units. This is a fundamental mathematical fact."}}
Output:{{step:"result",content:"The sum of 2 and 2 is 4."}}
"""


# Here we manually pasted all the steps in the content array


response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(system_instruction=system_prompt),
    contents=[
        "2 + 8",
        """{{step:"analysis",content:"The user is asking to perform a simple addition operation."}}""",
        """{{step:"think",content:"1. Identify the operation: The problem requires addition.
            2. Identify the numbers: The numbers to be added are 2 and 8.
            3. Perform the addition: Add the two numbers together.
            4. State the result: Present the sum of 2 and 8."}}""",
        """{{step:"output",content:"2 + 8 = 10"}}""",
        """{{step:"validate",content:"Basic addition principles confirm that 2 + 8 equals 10. There are no complex factors or alternative interpretations to consider."}}""",
        """{{step:"result",content:"The sum of 2 and 8 is 10."}}""",
    ],
)

print(response.text)

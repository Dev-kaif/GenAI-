from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

Gemini_key = os.getenv("GEMINI")

client = genai.Client(api_key=Gemini_key)


response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="i am kaif you are neko",  # gives system pre-defined instruction also known as system prompt
        max_output_tokens=500,  # gives max output limit of token
        temperature=0.8,  # adjust the creativity , more the float value more the creativity , less the float value more the accuracy and precision
    ),
    contents=["what is love?"],
)

print(response.text)


# how to send image:

#   response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents=[
#       types.Part.from_text('What is shown in this image?'),
#       types.Part.from_uri('gs://generativeai-downloads/images/scones.jpg',
#       'image/jpeg')
#     ]
#   )


# genrate the response in chunks not as whole para :

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=["Explain how AI works"]
# )
# for chunk in response:
#     print(chunk.text, end="")


# How to do chat :

# chat = client.chats.create(model="gemini-2.0-flash")

# # send invidual chat messages with send_message("String")
# chat.send_message("I have 2 dogs in my house.")

# # send another chat to continue the previous chat
# chat.send_message("How many paws are in my house?")

# # we can retive the chat histroy using chat.get_history() function
# for message in chat.get_history():
#     print(f"role : {message.role} , content : {message.parts[0].text}")

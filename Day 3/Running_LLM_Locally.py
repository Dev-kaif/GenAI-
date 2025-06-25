from ollama import Client

client = Client(host="http://localhost:3000")

my_Model = "your_model"

system_prompt = "your system prompt"

client.pull(my_Model)

while True:
    user = input("🧑 : ")

    if "end" in user:
        print("Ending the Chat ❌ ")
        break
    
    response = client.chat(
        model=my_Model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user},
        ],
    )

    print("👧 :", response["message"]["content"])

from together import Together
from collections import defaultdict

def build_receptionist_prompt(user_query: str) -> str:
    base_instruction = """
You are a receptionist working at a professional gym called OleeFit.
Your job is to welcome customers politely and help them with fitness-related queries.

You must detect and respond with the following:
1. Whether the customer's message is fitness-related.
2. Whether they mentioned their fitness level (Beginner, Intermediate, or Expert).
3. Whether they described a fitness problem (like "chest workout", "how to build abs", etc.).
4. Respond politely in a conversational tone, appropriate for a professional gym assistant.

Return your response in the following **JSON format** ONLY:
{
  "fitness_related": true or false,
  "exp_level": "Beginner" or "Intermediate" or "Expert" or "None",
  "fitness_problem": "user's described fitness issue or None",
  "response": "your polite reply to the customer here"
}

Example 1 (fitness-related, no level):
Input: "Can you give me a routine for building abs?"
Output:
{
  "fitness_related": true,
  "exp_level": "None",
  "fitness_problem": "routine for building abs",
  "response": "Thanks for your question! Before I connect you with a fitness instructor, could you tell me your current fitness level — Beginner, Intermediate, or Expert?"
}

Example 2 (not fitness-related):
Input: "What’s the stock market doing today?"
Output:
{
  "fitness_related": false,
  "exp_level": "None",
  "fitness_problem": "None",
  "response": "I'm here to assist with fitness-related questions only. Feel free to ask about workouts, training, or equipment!"
}

Example 3 (fitness-related with level):
Input: "I'm an intermediate lifter. Suggest exercises for chest."
Output:
{
  "fitness_related": true,
  "exp_level": "Intermediate",
  "fitness_problem": "exercises for chest",
  "response": "Thanks for sharing! Let me connect you with our fitness instructor for great chest exercises suited to your level."
}

Now evaluate the following input and return only the JSON.

Customer: """ + user_query

    return base_instruction


def general_query_llm(user_query: str, api_key: str,history: list) -> str:
    client = Together(api_key=api_key)

    prompt = build_receptionist_prompt(user_query)

    system_message = {
        "role": "system",
        "content": "You are a friendly and professional gym receptionist."
    }

    messages = [system_message] + history

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages
    )

    return response.choices[0].message.content.strip()


def build_fitness_prompt(user_query: str, exp_level: str, fitness_problem: str, retrieved_chunks: list) -> str:
    context = "\n\n".join([
        f"- {chunk['text']} (Body Part: {chunk['body_part']}, Equipment: {chunk['equipment']}, Level: {chunk['level']})"
        for chunk in retrieved_chunks
    ])

    prompt = f"""
You are a professional fitness assistant helping a {exp_level} level user.

The user has a fitness question: "{fitness_problem}"

Use the following information from our internal exercise database to provide a helpful, personalized answer.

Context:
{context}

Answer the user’s question in a clear, encouraging tone. Offer specific exercises or advice tailored to their level.

User: {user_query}
Assistant:"""

    return prompt.strip()



def call_fitness_llm(user_query: str,exp_level: str,fitness_problem:str, api_key: str,retrieved_chunks: list[dict]) -> str:
    client = Together(api_key=api_key)

    prompt = build_fitness_prompt(user_query, exp_level, fitness_problem, retrieved_chunks)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are a smart and friendly fitness assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content.strip()




from together import Together
from collections import defaultdict


def build_receptionist_prompt(user_query: str) -> str:
    base_instruction = """
You are a receptionist working at a professional gym called OleeFit.
Your job is to welcome customers politely and help them with fitness-related queries.

Rules:
1. If the customer says "hi", "hello", or similar, greet them and ask how you can help.
2. If the customer asks a fitness-related question (e.g., workouts, training plans, body parts, equipment), acknowledge and say you will connect them to a fitness instructor.
3. If the customer asks something unrelated to fitness (e.g., finance, movies, tech), politely decline and say you only handle fitness-related matters.
4. Always be respectful, brief, and professional.
"""

    prompt = f"{base_instruction}\n\nCustomer: {user_query}\nReceptionist:"
    return prompt


def general_query_llm(user_query: str, api_key: str) -> str:
    client = Together(api_key=api_key)

    prompt = build_receptionist_prompt(user_query)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly and professional gym receptionist."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
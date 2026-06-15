import os
from groq import Groq


def ask_question(query, index, chunks, model, k=10):

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding,
        k=min(k, len(chunks))
    )

    context = "\n\n------------------\n\n".join(
        [chunks[i] for i in indices[0]]
    )

    if "summary" in query.lower() or "summarize" in query.lower():
        prompt = f"""
You are a study assistant.

Read the content below and write a structured summary.

Format your response exactly like this:

## Main Topics
- List main topics

## Important Concepts
- List key concepts

## Key Points
- List important points

## Definitions
- Term: Definition

## Exam Notes
- List exam important points

Content:
{context}
"""

    elif "mcq" in query.lower() or "multiple choice" in query.lower():
        prompt = f"""
You are an exam paper setter.

Create 10 MCQ questions from the content below.
Each question must have exactly 4 options (A, B, C, D).
Do not provide answers.

Format:

Q1. Question?
A) Option
B) Option
C) Option
D) Option

Q2. Question?
A) Option
B) Option
C) Option
D) Option

Continue till Q10.

Content:
{context}
"""

    elif "important" in query.lower() or "exam question" in query.lower():
        prompt = f"""
You are an exam paper setter.

Generate 10 important exam questions from the content below.
Only questions. No answers. No options.

Format:

Q1. Question?

Q2. Question?

Continue till Q10.

Content:
{context}
"""

    else:
        prompt = f"""
You are a study assistant.
Answer the question using only the context below.
If the answer is not found in the context, respond with:
"This topic is not covered in the uploaded document."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000,
        temperature=0.3
    )

    return response.choices[0].message.content, context

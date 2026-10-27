
import os

from dotenv import load_dotenv
from groq import Groq

from app.generation.prompts import SYSTEM_PROMPT, build_prompt

load_dotenv()


class KnowledgeAssistantLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Add it to the project's .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    
    
    def generate_answer(self, question, documents):
        if not documents:
            return (
                "I couldn't find relevant information in the "
                "available documents. Please contact HR or "
                "upload the relevant policy document."
            )

        user_prompt = build_prompt(question, documents)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_completion_tokens=1200,
        )

        if not response.choices:
            return (
                "I couldn't generate an answer this time. "
                "Please try asking again."
            )

        message = response.choices[0].message
        answer = message.content

        if not isinstance(answer, str) or not answer.strip():
            finish_reason = response.choices[0].finish_reason
            print(
                "LLM returned an empty answer. "
                f"Finish reason: {finish_reason}"
            )
            return (
                "I couldn't generate a complete answer. "
                "Please try asking the question again."
            )

        return answer.strip()

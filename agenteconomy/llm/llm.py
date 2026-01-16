from dotenv import load_dotenv
load_dotenv()
from litellm import completion
import os
from openai import AsyncOpenAI
import asyncio 

class LLM:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

async def main():
    response = completion(
        model=os.getenv("MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response.choices[0].message.content)
    
if __name__ == "__main__":
    asyncio.run(main())
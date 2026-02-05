import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class LLMAgent:
    def __init__(self, name, prompt_file, reasoning_steps=0):
        self.name = name
        self.reasoning_steps = reasoning_steps
        with open(prompt_file, 'r') as f:
            self.prompt = f.read()
        self.history = []


    def act(self, game_history):
        prompt = self.build_prompt(game_history)
        response = self.query_nvidia_llm(prompt)
        action = self.parse_action(response)
        self.last_reasoning = response
        return action

    def build_prompt(self, game_history):
        history_text = ""
        for i, (a1, a2, p1, p2) in enumerate(game_history):
            history_text += f"Round {i+1}: You={a1}, Opponent={a2}\n"

        full_prompt = self.prompt + "\nGame History:\n" + history_text

        if self.reasoning_steps > 0:
            full_prompt += f"""
    Think step by step for exactly {self.reasoning_steps} reasoning steps.
    Number each step (Step 1, Step 2, ...).
    After reasoning, output ONLY one character: C or D.
    """
        else:
            full_prompt += "\nChoose your next action: C or D only."

        return full_prompt


    def query_nvidia_llm(self, prompt):
        
        client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv("NVIDIA_API_KEY")
        )

        completion = client.chat.completions.create(
        #model="meta/llama-3.1-405b-instruct",
        model="google/gemma-3-27b-it",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
        )

        response_text = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_text += content
        
        return response_text


    def parse_action(self, text):
        text = text.strip().upper()
        last_char = text[-1]

        if last_char == "C":
            return "C"
        elif last_char == "D":
            return "D"
        else:
            return "C"  # safe fallback




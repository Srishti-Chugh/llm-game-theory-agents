# agents/llm_agent_quantum.py

import os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class QuantumLLMAgent:
    def __init__(self, name, prompt_file):
        self.name = name
        with open(prompt_file, "r") as f:
            self.prompt = f.read()

    def act(self, history):
        prompt = self.build_prompt(history)
        response = self.query_llm(prompt)
        return self.parse_theta(response)

    def build_prompt(self, history):
        text = self.prompt + "\nHistory:\n"
        for i,(t1,t2,out,p1,p2) in enumerate(history):
            text += f"Round {i+1}: You={t1:.2f}, Opponent={t2:.2f}, Outcome={out}, Payoff={p1}\n"
        text += "\nOutput one number theta between 0 and 1.57."
        return text

    def query_llm(self, prompt):
        client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )

        completion = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        return completion.choices[0].message.content

    def parse_theta(self, text):
        nums = re.findall(r"\d+\.?\d*", text)
        if not nums:
            return 0.5
        theta = float(nums[0])
        return max(0.0, min(theta, 1.57))
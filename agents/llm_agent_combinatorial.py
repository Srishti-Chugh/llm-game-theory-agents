# agents/llm_agent_combinatorial.py

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class CombinatorialLLMAgent:
    def __init__(self, name, prompt_file, num_items=4, total_budget=100):
        self.name = name
        self.num_items = num_items
        self.total_budget = total_budget

        with open(prompt_file, "r") as f:
            self.prompt = f.read()

    def act(self, history):
        prompt = self.build_prompt(history)
        response = self.query_llm(prompt)
        allocation = self.parse_allocation(response)
        return allocation

    def build_prompt(self, history):
        text = self.prompt + "\n\nHistory:\n"
        for i, (a1, a2, p1, p2) in enumerate(history):
            text += f"Round {i+1}: You={a1}, Opponent={a2}, Payoff={p1}\n"

        text += (
            f"\nOutput ONLY a list of {self.num_items} integers "
            f"that sum to {self.total_budget}.\n"
            f"Example: 25,25,25,25\n"
            f"No explanation. No text."
        )
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
            max_tokens=128
        )

        return completion.choices[0].message.content.strip()

    def parse_allocation(self, text):
        """
        Parse allocation safely and enforce fixed dimension + normalization.
        """

        # Extract only first line
        line = text.split("\n")[0]

        # Extract numbers
        nums = re.findall(r"-?\d+\.?\d*", line)
        nums = [float(x) for x in nums]

        # If nothing valid, fallback to uniform
        if len(nums) == 0:
            return self.default_allocation()

        # Trim or pad
        if len(nums) > self.num_items:
            nums = nums[:self.num_items]
        elif len(nums) < self.num_items:
            nums = nums + [0.0] * (self.num_items - len(nums))

        arr = np.array(nums)
        arr[arr < 0] = 0

        # Normalize
        if arr.sum() == 0:
            arr = np.ones(self.num_items) * (self.total_budget / self.num_items)
        else:
            arr = (arr / arr.sum()) * self.total_budget

        return arr.round(2).tolist()

    def default_allocation(self):
        return [self.total_budget / self.num_items] * self.num_items
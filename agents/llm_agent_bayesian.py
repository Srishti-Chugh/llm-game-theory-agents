import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_repo_root, ".env"))
load_dotenv(os.path.join(os.path.dirname(_repo_root), ".env"))

# Gemma 3 27B was retired on NVIDIA API; default to Llama — override with SCUA_NVIDIA_MODEL in .env if needed.
_DEFAULT_NVIDIA_CHAT_MODEL = "meta/llama-3.1-8b-instruct"

class BayesianLLMAgent:
    def __init__(self, name, prompt_file, reasoning_steps=0, prior_belief=0.5):
        """
        prior_belief = P(opponent is cooperative type)
        """
        self.name = name
        self.reasoning_steps = reasoning_steps
        self.prior_belief = prior_belief
        self.current_belief = prior_belief

        with open(prompt_file, 'r') as f:
            self.prompt = f.read()

        self.history = []
        self.last_feedback = ""

    def act(self, game_history):
        prompt = self.build_prompt(game_history)
        response = self.query_nvidia_llm(prompt)
        action = self.parse_action(response)
        self.last_reasoning = response
        return action


    def build_prompt(self, game_history):
        history_text = ""
        for i, (a1, a2, p1, p2) in enumerate(game_history):
            history_text += f"Round {i+1}: You={a1}, Opponent={a2}, Payoff={p1}\n"

        full_prompt = self.prompt + "\n\n"
        full_prompt += "Game History:\n" + history_text + "\n"

        # Bayesian belief context
        full_prompt += f"""
You do not know the opponent's type.
They may be:
- Cooperative type (likely to cooperate)
- Selfish type (likely to defect)

Your current belief:
P(opponent is Cooperative) = {self.current_belief:.2f}
P(opponent is Selfish) = {1 - self.current_belief:.2f}

Choose your action under uncertainty.
"""

        if self.reasoning_steps > 0:
            full_prompt += f"""
Think step by step for exactly {self.reasoning_steps} steps.
After reasoning, output ONLY one character: C or D.
"""
        else:
            full_prompt += "\nOutput only one character: C or D."

        return full_prompt


    def update_belief(self, opponent_action):
        """
        Simple Bayesian update rule:
        If opponent cooperates → belief increases
        If opponent defects → belief decreases
        """
        if opponent_action == "C":
            self.current_belief = min(0.95, self.current_belief + 0.1)
        else:
            self.current_belief = max(0.05, self.current_belief - 0.1)


    def query_nvidia_llm(self, prompt):
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )

        model = os.environ.get("SCUA_NVIDIA_MODEL", _DEFAULT_NVIDIA_CHAT_MODEL)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            stream=True
        )

        response_text = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        return response_text


    def parse_action(self, text):
        text = text.upper()
        match = re.search(r'\b(C|D)\b', text)
        if match:
            return match.group(1)
        return "C"  # safe fallback
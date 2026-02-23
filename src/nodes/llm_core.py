import yaml
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.state import SCUAState

load_dotenv()

def llm_executive_node(state: SCUAState):
    with open("src/prompts/templates.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    
    # Initialize with original parameters: Temp 0.0 and specific Model
    llm = ChatNVIDIA(
        model="google/gemma-3-27b-it", # Matching your original model
        temperature=0.0,               # Ensures deterministic strategic moves
        max_tokens=500                 # Matching your original constraint
    )
    
    prompt_text = prompts['synthesis_packet'].format(
        game_regime=state['game_regime'],
        opponent_profile="Persistent Observer",
        entropy=state['metrics'].get('entropy', 0),
        volatility=state['metrics'].get('volatility', 0),
        nash_dev=state['metrics'].get('nash_dev', 0),
        p_deceptive=state['beliefs'].get('Deceptive', 0),
        p_noisy=state['beliefs'].get('Noisy', 0),
        p_non_stationary=state['beliefs'].get('Non-Stationary', 0),
        p_irrational=state['beliefs'].get('Irrational', 0),
        logic_recommendation=state['logic_recommendation']
    )
    
    # .invoke() replaces .chat.completions.create() in LangChain
    response = llm.invoke(prompt_text) 
    
    return {"final_action": response.content, "messages": [response]}

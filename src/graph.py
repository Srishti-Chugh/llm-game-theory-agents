import sqlite3

from langgraph.graph import StateGraph, END
from src.state import SCUAState
from src.nodes.perception import perception_node
from src.nodes.bayesian import bayesian_engine_node
from src.nodes.logic import logic_layer_node
from src.nodes.llm_core import llm_executive_node

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

def create_scua_graph():
    conn = sqlite3.connect("scua_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    workflow = StateGraph(SCUAState)
    
    # Add Nodes
    workflow.add_node("perception", perception_node)
    workflow.add_node("bayesian_engine", bayesian_engine_node)
    workflow.add_node("logic_layer", logic_layer_node)
    workflow.add_node("executive_function", llm_executive_node)

    # Define Edges (The Flow)
    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "bayesian_engine")
    workflow.add_edge("bayesian_engine", "logic_layer")
    workflow.add_edge("logic_layer", "executive_function")
    workflow.add_edge("executive_function", END)

    return workflow.compile(checkpointer=memory) 

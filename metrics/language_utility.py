def language_feedback(action, opponent_action):
    if action == "D" and opponent_action == "C":
        return "You betrayed your opponent. This is morally wrong."
    elif action == "C" and opponent_action == "C":
        return "You cooperated and built mutual trust."
    elif action == "D" and opponent_action == "D":
        return "Both chose selfish actions."
    else:
        return "You tried to cooperate but were exploited."

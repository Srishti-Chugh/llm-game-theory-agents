def expected_utility(belief_coop, action):
    """
    Prisoner's Dilemma payoffs:
    R=3, T=5, S=0, P=1
    """
    R, T, S, P = 3, 5, 0, 1
    belief_selfish = 1 - belief_coop

    if action == "C":
        return belief_coop * R + belief_selfish * S
    else:
        return belief_coop * T + belief_selfish * P


def optimal_action(belief_coop):
    eu_c = expected_utility(belief_coop, "C")
    eu_d = expected_utility(belief_coop, "D")
    return "C" if eu_c >= eu_d else "D"


def regret(belief_coop, chosen_action):
    eu_opt = max(expected_utility(belief_coop,"C"),
                 expected_utility(belief_coop,"D"))
    eu_chosen = expected_utility(belief_coop, chosen_action)
    return eu_opt - eu_chosen
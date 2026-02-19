def cooperation_rate(actions):
    return actions.count("C") / len(actions)

def mutual_cooperation_rate(actions1, actions2):
    count = 0
    for a1, a2 in zip(actions1, actions2):
        if a1 == "C" and a2 == "C":
            count += 1
    return count / len(actions1)

def reciprocity(actions1, actions2):
    coop_after_coop = 0
    coop_after_defect = 0
    total_after_coop = 0
    total_after_defect = 0

    for i in range(1, len(actions1)):
        if actions2[i-1] == "C":
            total_after_coop += 1
            if actions1[i] == "C":
                coop_after_coop += 1
        if actions2[i-1] == "D":
            total_after_defect += 1
            if actions1[i] == "C":
                coop_after_defect += 1

    p_cc = coop_after_coop / total_after_coop if total_after_coop else 0
    p_cd = coop_after_defect / total_after_defect if total_after_defect else 0

    return p_cc, p_cd

def nash_deviation(actions_a, actions_b, ne=("D","D")):
    count = 0
    for a,b in zip(actions_a, actions_b):
        if (a,b) != ne:
            count += 1
    return count / len(actions_a)

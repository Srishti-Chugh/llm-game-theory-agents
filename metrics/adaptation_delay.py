def adaptation_delay(actions, change_point):
    baseline = actions[:change_point]
    after = actions[change_point:]

    base_mean = sum(baseline) / len(baseline)

    for i, a in enumerate(after):
        if abs(a - base_mean) > 0.1:
            return i + 1

    return len(after)
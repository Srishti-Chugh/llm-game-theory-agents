# metrics/collapse_instability.py

def collapse_instability(actions, measurements):
    """
    actions: agent actions
    measurements: observed quantum outcomes (0/1)

    Instability when action flips after measurement change.
    """
    if len(actions) < 2:
        return 0.0

    unstable = 0
    total = 0

    for i in range(1, len(actions)):
        if measurements[i] != measurements[i-1]:
            total += 1
            if actions[i] != actions[i-1]:
                unstable += 1

    if total == 0:
        return 0.0

    return unstable / total
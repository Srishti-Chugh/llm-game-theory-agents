import matplotlib.pyplot as plt

def plot_entropy(entropy_a, entropy_b):
    plt.plot(entropy_a, label="Agent A")
    plt.plot(entropy_b, label="Agent B")
    plt.xlabel("Round")
    plt.ylabel("Entropy")
    plt.title("Entropy Over Time")
    plt.legend()
    plt.show()


def plot_cooperation(actions_a, actions_b):
    coop_a = [1 if a=="C" else 0 for a in actions_a]
    coop_b = [1 if b=="C" else 0 for b in actions_b]

    plt.plot(coop_a, label="Agent A")
    plt.plot(coop_b, label="Agent B")
    plt.xlabel("Round")
    plt.ylabel("Cooperate (1) / Defect (0)")
    plt.title("Cooperation Over Time")
    plt.legend()
    plt.show()

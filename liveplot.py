import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self):
        self.reward = []
        self.episodes = []
        self.q_loss = []
        self.reward_avgs = []
        plt.ioff()

    def append(self, x, reward, q_loss):
        self.episodes.append(x)
        self.reward.append(reward)
        self.q_loss.append(q_loss)

        plt.figure(1)
        plt.xlim(-0.01, max(self.episodes))
        plt.ylim(min(self.reward), max(self.reward) + 0.01)

        plt.scatter(self.episodes[-1:], self.reward[-1:], color="#000000")

        if len(self.reward) > 100:
            self.reward_avgs.append(sum(self.reward[-100:]) / 100)
            if len(self.reward_avgs) > 1:
                plt.plot(self.episodes[-2:], self.reward_avgs[-2:], color="#ffa500")

        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("$R_1$")

        plt.draw()

        plt.figure(2)
        plt.xlim(-0.01, max(self.episodes))
        plt.ylim(-0.01, max(self.q_loss))

        plt.plot(self.episodes[-2:], self.q_loss[-2:], color="#0000ff")
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Q-Loss")

        plt.draw()

    def draw(self, block=False):
        plt.pause(0.1)
        plt.show(block=block)

    def save(self):
        with open("q_loss.log", "w") as f:
            f.write(str(self.q_loss))
            f.close()

        with open("reward.log", "w") as f:
            f.write(str(self.reward))
            f.close()

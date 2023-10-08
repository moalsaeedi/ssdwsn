
import numpy as np
import matplotlib.pyplot as plt

def polyak_average(net, target_net, tau=0.01):
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

def plot_learning_curve(x, scores, figure_file, title, idx):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.subplot(4, 4, idx)
    plt.plot(x, running_avg)
    plt.title(title)
    plt.savefig(figure_file)
 
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.zeros(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count /tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
import numpy as np
import pickle
from IPython import embed

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = 1e-4
        self.epsilon = 1e-8
        self.clip = 10
        self.maxCount = 4096
        self.curCount = 0

    # update mean and var with current input
    def update(self, x):
        # embed()
        # exit(0)
        if self.curCount > self.maxCount:
            return
        # print("@")
        x_ = x.reshape(-1, len(self.mean))
        batch_mean = np.mean(x_, axis=0)
        batch_var = np.var(x_, axis=0)
        batch_count = x_.shape[0]
        self.curCount += 1
        # embed()
        # exit(0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    # get value from normalized output
    def apply(self, x):
        # embed()
        # exit(0)
        self.update(x)
        x = np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
        return x

    def applyOnly(self, x):
        # embed()
        # exit(0)
        x = np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
        return x

        # rms_x = np.zeros(shape=x.shape(), np.float32)
        # for i in range(len(rms_x)):
        #     rms_x[i] = np.clip((x[i] - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)


    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def save(self, path):
        data = {'mean':self.mean, 'var':self.var, 'count':self.count}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
                data = pickle.load(f)
                self.mean = data['mean']
                self.var = data['var']
                self.count = data['count']

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from errors import BS, NLL, accuracy

# taken and modified from TODO
class TemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS", loss='NLL'):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.loss = loss

    def _loss_fun(self, x, logits, true):
        scaled_l = self.predict(logits, x)
        if self.loss == 'BS':
            loss = BS(scaled_l, true)
        elif self.loss == 'NLL':
            loss = NLL(scaled_l, true)
        return loss

    # Find the temperature
    def fit(self, logits, true, verbose=False):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]

        if verbose:
            print("Temperature:", 1/self.temp)

        return opt

    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return logits/self.temp
        else:
            return logits/temp


# inverse of softmax
# https://en.wikipedia.org/wiki/Logit-normal_distribution
def logistic_func(probs):
    """
    >>> probs = np.array([[0.1,0.9], [0.5,0.5]])
    >>> logs = logistic_func(probs)
    >>> from scipy.special import softmax
    >>> softmax(logs, axis=1)
    array([[0.1, 0.9],
           [0.5, 0.5]])
    """
    logits = np.zeros_like(probs)
    n = probs.shape[0] - 1
    logits[:, :-1] = (np.log(probs[:, :-1]).transpose() - np.log(probs[:, -1]).transpose()).transpose()
    return logits


def mse_t(t, *args):
    # find optimal temperature with MSE loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)
    p = np.exp(logit)/n[:,None]
    mse = np.mean((p-label)**2)
    return mse


def ll_t(t, *args):
    # find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)
    p = np.clip(np.exp(logit)/n[:,None],1e-20,1-1e-20)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce


def mse_w(w, *args):
    # find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0]*p0+w[1]*p1+w[2]*p2
    p = p/np.sum(p,1)[:,None]
    mse = np.mean((p-label)**2)
    return mse


def ll_w(w, *args):
    # find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0]*p0+w[1]*p1+w[2]*p2)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce


# taken and modified from
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/util_calibration.py
class ETScaling():

    def __init__(self, w=None, t=None):
        self.w = w
        self.t = t

    # Find the temperature
    def fit(self, logits, label, loss='mse'):

        bnds = ((0.05, 5.0),)
        label = label_binarize(np.array(label), classes=np.unique(np.array(label)))
        if loss == 'ce':
            t = minimize(
                ll_t, 1.0, args=(logits, label), method='L-BFGS-B',
                bounds=bnds, tol=1e-12)
        if loss == 'mse':
            t = minimize(
                mse_t, 1.0, args=(logits, label), method='L-BFGS-B',
                bounds=bnds, tol=1e-12)
        self.t = t.x

        n_class = logits.shape[1]
        p1 = softmax(logits, axis=1)
        logits = logits/self.t
        p0 = softmax(logits, axis=1)
        p2 = np.ones_like(p0)/n_class

        bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = {"type": "eq", "fun": my_constraint_fun,}
        if loss == 'ce':
            w = minimize(
                ll_w, (1.0, 0.0, 0.0), args=(p0,p1,p2,label), method='SLSQP',
                constraints = constraints, bounds=bnds_w, tol=1e-12,
                options={'disp': True})
        if loss == 'mse':
            w = minimize(
                mse_w, (1.0, 0.0, 0.0), args=(p0,p1,p2,label),
                method='SLSQP', constraints = constraints, bounds=bnds_w,
                tol=1e-12, options={'disp': True})
        self.w = w.x

    def predict(self, logits):
        n_class = logits.shape[1]
        p1 = softmax(logits, axis=1)
        logits = logits/self.t
        p0 = softmax(logits, axis=1)
        p2 = np.ones_like(p0)/n_class
        p = self.w[0]*p0 + self.w[1]*p1 + self.w[2]*p2
        # return to logits for consistency with the errors
        return logistic_func(p)


class FlawedRecal():

    def __init__(self, acc=None):
        self.acc = acc

    def fit(self, logits, labels):
        self.acc = accuracy(logits, labels)

    def predict(self, logits):
        arg = logits.argmax(-1)
        probs = np.zeros(logits.shape) + (1-self.acc)/(logits.shape[1]-1)
        probs[np.arange(logits.shape[0]), arg] = self.acc
        return logistic_func(probs)

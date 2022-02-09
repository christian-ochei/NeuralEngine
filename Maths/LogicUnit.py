import numpy as np

def reduce_mean(x):
    return np.mean(x)


def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])


def objective(x, y):
	return x**2.0 + y**2.0

def relu(x,derivative=False):
    if derivative==True:
        mask = (x > 0) * 1.0
    mask = (x>0) * 1.0
    return mask*x

def arctan(x,derivative=False):
    if derivative:
        # return np.arctan(x)
        # return np.arctan(x)
        return 1 / (1 + x ** 2)
    return np.arctan(x)


def log(x,derivative=False):
    if derivative:
        return log(x) * (1 - log(x))
    return 1 / ( 1+ np.exp(-1*x))

def tanh(x,derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return  np.tanh(x)

# calculate cross entropy
def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

# print(cross_entropy(np.array([20 for x in range(2)]).astype('float64'),np.array([0 for x in range(2)]).astype('float64')))

def softmax(x, derivative=False) -> float:
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

def sigmoid(x, derivative=False) -> float:
    if derivative:
        # return x
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return (1 / (1 + np.exp(-x)))

def linear(x,derivative=False) ->float:
    if derivative:
        x[:] = 1
        x[x<0] = -1
        return x
    return x

def cliping(x,derivative=False):
    if derivative:
        return np.where(a:=np.where(x,x>5,-1e-4),a<-5,1e-4)
    return np.clip(x,-5,5)

# def square(x,derivative=False):
#     if derivative:
#         return x*2
#     return (x/100)**2
def sin(x,derivative=False):
    if derivative:
        return np.cos(x)
    return np.sin(x)

def MSE(Y, YH):
    return np.square(Y - YH).mean()


def as_numpy(x):
    return np.array(x).astype('float64')

def as_numpy_nocast(x):
    return np.array(x)

def clip(a,b,c):
    return min(max(b,a),c)



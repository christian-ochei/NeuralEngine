from . Maths.LogicUnit import *
from . Model_utils import *
from . Layer import Layer
from . import Errors
from random import uniform
import numpy as np
from . Model_utils import CoAdverserial
from math import sqrt

from random import randint
c = [3]
class Batch:
    def __init__(s,*models):
        s.models = models

    def __enter__(s):
        for model in s.models:
            model._batch = True

    def __exit__(s, exc_type, exc_val, exc_tb):
        for model in s.models:
            model.train_count += 1
            m = len(model._avg_weights)
            model._batch = False
            for d_wb in model._avg_weights:
                model.update_w_and_b((d_wb,None),div=m)
            model._avg_weights = []




class Adverserial:
    def __init__(s,*models):
        s.models = models

    def __enter__(s):
        for model in s.models:
            model._adverserial = True

    def __exit__(s, exc_type, exc_val, exc_tb):
        for model in s.models:
            model._adverserial = False

class Network:
    def __init__(s,layers,Type=...):
        # This initializes parameters

        if layers:
            assert layers[0].__class__  is Layer.Input or layers[0].__class__ is Layer.DynamicInput

            if layers[0].__class__ is Layer.DynamicInput and len(layers)>1:
                layers[0](layers[1])

            for x in range(len(layers)-1):
                if layers[x+1].__class__ is Layer.DynamicConnection:
                    layers[x+1](layers[x],layers[x+2] if x<len(layers)-2 else None)
                else:
                    layers[x+1](layers[x])

        s._layers:List[Layer.Dense] = layers
        s._adverserial = False

        if layers:
            s.loss = None
            s.input_size = layers[0].size
            s.output_size = layers[len(layers)-1].size
            s.d_wb_buffer = []
            s.alpha = 0.1
            s.type = Type
            s.steps = 1
            s._param_buffer = []
            s.gradient_tape = False
            s._batch = False
            s.exponent_gradients = 1.2

            s._avg_weights = []
            s.train_count = 0
            s.i_div       = 1
            s.dynamic_layers = [layer for layer in s._layers if layer.__class__ is Layer.DynamicConnection]

            # print(s.dynamic_layers)
            # print(len(s.dynamic_layers))
            # print(len(s._layers))
            # fsdfs

    def randomize(s,extent):

        for weight,bias in [(layer.weights,layer.bias) for layer in s._layers[1:]]:
            weight += np.random.uniform(-extent,extent,weight.shape)
            bias   += uniform(-extent,extent)

    def safe_add_nodes(s,mass=10):
        for layer in s.dynamic_layers:
            if randint(0,2) == 1:
                layer.__add__(randint(1,mass))

    def safe_delete_nodes(s,mass=5):
        for layer in s.dynamic_layers:
            # print(layer.size)
            if layer.size > 10 and randint(0, 2) == 1:
                layer.__delitem__(randint(1, mass))

    def set_nodes(s,model):
        for layer,o_layer in zip(s.dynamic_layers,model.dynamic_layers):
            if len(layer) < len(o_layer):
                layer.__add__(abs(len(o_layer)-len(layer)))
            elif len(layer) != len(o_layer):
                layer.__delitem__(abs(len(layer)-len(o_layer)))



    def feed_forward(s,x_train):
        if not x_train.size == s._layers[0].size:
            raise IndexError(f"Input Layer and x_prediction must be the same size {x_train.size} != model.input_size:{s._layers[0].size}")

        # first item will always be x_train then callbacks
        params    = [Callback(None,None,x_train)]
        activated = x_train

        for layer in s._layers[1:]:

            activated = layer.forward(activated,params)

        return params,activated


    def feed_backwards(s,Y,output,function=None):
        params,out = output
        scores     = out
        d_wb       = []
        N          = params[0].activated.shape

        scores_max = np.max(scores,axis=1,keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        prob       = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)

        correct_logprobs = -np.log(probs[range(N)]
                                   , Y)

        data_loss = np.sum(correct_logprobs)/N

        dscores   = probs

        dscores[range(N),Y] -= 1
        dscores /= N
        grad     = []
        prod     = lambda:np.dot(params[-x].activated.T,dscores)

        for x in range(1,len(s._layers)-1):
            dw = np.dot()
            db = np.sum(dscores,axis=0,keepdims=True)

            errors.append(error)
            d_wb.append(
                (dw:=np.outer(error, params[last-1-x].activated),
                 np.sum(error)
                 )
            )
            dh = np.dot(dscores,W2)
            # prod = lambda:np.dot(s._layers[last-x+1].weights.T,error)
            prod = lambda:np.dot(params[last-x].activated.T,error)

        return d_wb,errors

    # def backwards(s, Y, output, function=None):
    #     # global [c]
    #     params,out = output
    #     # c[0] -= 0.0001
    #
    #
    #
    #     # loss = (Y-out)-(params[-1].biased)
    #     # lossdiv = Y/params[-1].dotted
    #     # # print(Y-out,Y,out)
    #     # v = s._layers[-1].weights.T[:]
    #     #
    #     # wd = (v/(v*lossdiv))
    #     # # print(s._layers[-1].weights.T[:].shape, 'b')
    #     #
    #     # s._layers[-1].weights.T[:] *= lossdiv
    #     # s._layers[-1].bias += loss
    #
    #     # s._layers[-2].weights.T[:] *= wd
    #
    #     # print(s._layers[-2].bias, 's')
    #
    #     #
    #
    #     # print(s._layers[-1].weights.T[:].shape,'a')
    #     # print(c)
    #
    #
    #
    #     # s._layers[-1].bias += loss
    #     # print(s._layers[-1].weights.T.shape)

    #
    # def batch_train(s,X,Y):
    #     DATA = [s.feed_forward(X[x],Y[x]) for x in range(len(X))]
    #
    #




    # def Adam(s,derivative):
    #     ...
    #
    #     x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    #     score = objective(x[0], x[1])
    #
    #     m = np.zeros(bounds.shape[0])
    #     v = np.zeros(bounds.shape[0])
    #
    #     for t in range(n_iter):
    #         g = derivative(x[0], x[1])
    #         for i in range(x.shape[0]):
    #             # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
    #             m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    #             # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
    #             v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
    #             # mhat(t) = m(t) / (1 - beta1(t))
    #             mhat = m[i] / (1.0 - beta1 ** (t + 1))
    #             # vhat(t) = v(t) / (1 - beta2(t))
    #             vhat = v[i] / (1.0 - beta2 ** (t + 1))
    #             # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
    #             x[i] = x[i] - alpha * mhat / (np.sqrt(vhat) + eps)
    #         # evaluate candidate point
    #         score = objective(x[0], x[1])
    #         # report progress
    #         print('>%d f(%s) = %.5f' % (t, x, score))
    #     return [x, score]
    #
    #


    def update_w_and_b(s,d_wb,div=1):
        ...
        d_wb,errors = d_wb
        # s.alpha = 0.1

        if not s._batch:
            s.steps += 1
            if s.alpha is None or s.alpha is ...:
                alpha = 0.001
            else:
                alpha = s.alpha


            # print(alpha)
            # alpha = 2
            clip = 100

            # a = 1/(abs(s.steps)**0.2)
            a = s.alpha
            # alpha = 1
            # print(s.alpha)
            # a = 1

            noi =  -a,a*4
            # noi =  1,1
            # noi = 1,1
            b_epsilon = uniform(*noi)
            for layer,(d_weight,d_bias) in zip(reversed(s._layers),d_wb):
                ...
                if not layer.__class__ == Layer.N:
                    epsilon = np.random.uniform(*noi, layer.weights.shape)
                    layer.weights += (v:=np.clip((d_weight/div)*alpha*epsilon,  -clip,clip))*10
                    layer.bias    += (v:=np.clip((d_bias  /div)*alpha*b_epsilon,-clip,clip))*10
        else:
            s._avg_weights.append(d_wb)



    def __getitem__(s, x_train):
        if s._adverserial:
            return CoAdverserial(s,x_train)
        return s.feed_forward(as_numpy(x_train))[1]

    def __call__(s,x_train):
        return s[x_train]

    def __setitem__(s, x_train,y_train):
        ...
        if not isinstance(y_train,TrainStepToken):
            ...
            # print(randint(0,12))
            # print(as_numpy(y_train),as_numpy(y_train))
            s.update_w_and_b(s.feed_backwards(as_numpy(y_train)/s.i_div,s.feed_forward(as_numpy(x_train))))

    def __repr__(s):
        return str('<Network>')

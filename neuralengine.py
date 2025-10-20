from Maths.LogicUnit import *
from typing import List
from Model_utils import *
from Layer import Layer
import Errors
from random import uniform
import numpy as np
from Model_utils import CoAdverserial
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
            s.epsilon = 1
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

    def randomize_weights(s,extent):

        for weight,bias in [(layer.weights,layer.bias) for layer in s._layers[1:]]:
            weight += np.random.uniform(-extent,extent,weight.shape)
            bias   += np.random.uniform(-extent,extent,bias.shape)

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

    def forward(s,x_train):
        if not x_train.size == s._layers[0].size:
            raise IndexError(f"Input Layer and input X must be the same size {x_train.size} != model.input_size:{s._layers[0].size}")

        params    = [Callback(None,None,x_train)]
        activated = x_train

        for layer in s._layers[1:]:
            activated = layer.forward(activated,params)

        return params,activated

    def backward(s, Y, output, function=None):
        # global [c]
        n_batch    = 1
        params,out = output

        if out.shape != Y.shape:
            raise ValueError(f"out.shape != Y.shape : {out.shape} != {Y.shape}")

        d = out-Y

        grad = []

        for x,layer in enumerate(s._layers[:1:-1]):

            dw = (1./n_batch) * np.matmul(d,params[-1-x].activated.T)
            # db = (1./n_batch) * np.sum(d,keepdims=True)
            dw = np.zeros(layer.weights.shape)
            db = np.zeros(d.shape)


            grad.append([dw,db])

            da = np.matmul(s._layers[-x-1].weights.T,d)
            dz = da * s._layers[-x-1].activation(params[-x-2].dotted,derivative=True)
            # dz = 0
            # dz = da * -(params[-x-2].activated/params[-x-2].biased)
            # dz = da * -(params[-x-2].activated/params[-x-2].activated)

            d = da
            # d = params[-x-1].biased
            # break
        return grad


    def update(s,d_wb,div=1):
        ...
        # d_wb,errors = d_wb
        # s.alpha = 0.1

        if not s._batch:
            s.steps += 1
            if s.alpha is None or s.alpha is ...:
                alpha = 0.001
            else:
                alpha = s.alpha

            a = s.epsilon
            clip = 100
            beta = .9
            noi  =  -a,a*4

            for layer,(d_weight,d_bias) in zip(reversed(s._layers),d_wb):
                ...
                if not layer.__class__ == Layer.N:
                    epsilon   = np.random.uniform(*noi,layer.weights.shape)
                    b_epsilon = np.random.uniform(*noi,layer.bias.shape)

                    layer.weights +=np.clip( d_weight*alpha/4,-8,8)
                    layer.bias +=   np.clip(d_bias*alpha/4,-8,8)

                    # dw = (beta * 0 + (1-beta) * d_weight)
                    # db = (beta * 0 + (1-beta) * d_bias)
                    #
                    # layer.weights -= alpha * dw * epsilon
                    # layer.bias    -= alpha * db * b_epsilon
                    # print('s')
                    # epsilon = np.random.uniform(*noi, layer.weights.shape)
                    # layer.weights += (v:=np.clip((d_weight/div)*alpha*epsilon,  -clip,clip))*10
                    # layer.bias    += (v:=np.clip((d_bias  /div)*alpha*b_epsilon,-clip,clip))*10
        else:
            s._avg_weights.append(d_wb)




    def __getitem__(s, x_train):
        if s._adverserial:
            return CoAdverserial(s,x_train)
        return s.forward(as_numpy(x_train))[1]

    def __call__(s,x_train):
        return s[x_train]

    def __setitem__(s, x_train,y_train):
        ...
        if not isinstance(y_train,TrainStepToken):
            ...
            # print(randint(0,12))
            # print(as_numpy(y_train),as_numpy(y_train))
            s.update(s.backward(as_numpy(y_train)/s.i_div,s.forward(as_numpy(x_train))))

    def __repr__(s):
        return str('<Network>')

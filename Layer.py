from Maths.LogicUnit import *
from Model_utils import Callback
import traceback
from random import uniform

class Layer:
    class LayerUtils:
        __layer_counter = 0
        def __init__(s, size):
            tb = traceback.format_stack()
            tb = tb[:len(tb) - 2]
            s.traceback = ''.join(tb)


            s.name = str(s.__class__.__name__)+str(Layer.LayerUtils.__layer_counter)
            s.type = s.__class__
            s.size = size
            Layer.LayerUtils.__layer_counter += 1

        def __len__(s):
            return s.size

    class Input(LayerUtils):
        def __init__(s,size):
            Layer.LayerUtils.__init__(s,size)
            s.input_buffer = np.zeros(size)

    class DynamicInput(LayerUtils):
        def __init__(s,size):
            Layer.LayerUtils.__init__(s,size)
            s.input_buffer = np.zeros(size)
            s.next_layer   = ...

        def __call__(s,next_layer):
            s.next_layer = next_layer

        def __delitem__(s, mass:int):
            assert mass > 0
            s.size -= mass
            s.next_layer.weights = np.delete(s.next_layer.weights, mass - 1, axis=1)
            s.next_layer.size = s.next_layer.weights.shape[0]

        def __add__(s, mass:int):
            assert mass > 0
            nwx = (np.random.random((s.next_layer.size, other)) * np.sqrt(1. / s.size)).astype(np.float64)
            s.next_layer.weights = np.hstack([s.next_layer.weights, nwx])
            s.size += mass
            s.next_layer.size = s.next_layer.weights.shape[0]
            return s

    class Dense(LayerUtils):
        def __init__(s, size,activation=linear):
            super().__init__(size)
            s.weights = ...
            s.bias    = ...
            s.prev_layer = ...
            s.activation = activation

        def __call__(s,prev_layer):
            s.weights = (np.random.uniform(-1,1,(s.size,len(prev_layer))) * np.sqrt(1./s.size)).astype(np.float64)
            s.bias    = np.random.uniform(-1,1,s.size)
            return s

        def forward(s,activated,params):
            dotted = np.dot(s.weights, activated)
            biased = dotted+s.bias
            activated = s.activation(biased)
            params.append(Callback(dotted,biased,activated))
            return activated

    class N(LayerUtils):
        def __init__(s, size, activation=linear):
            super().__init__(size)
            s.activation = activation
            ...
        def __call__(s,_):
            ...

        def forward(s,activated,params):
            dotted = activated
            biased = activated
            activated = s.activation(activated)
            params.append(Callback(dotted,biased,activated))
            return activated



    class DynamicConnection(LayerUtils):
        def __init__(s, size,activation=linear):
            super().__init__(size)
            s.weights = ...
            s.bias    = ...
            s.prev_layer = ...
            s.activation = activation
            s.next_layer = ...

        def __call__(s,prev_layer,next_layer):
            s.prev_layer = prev_layer
            s.next_layer = next_layer
            s.weights = (np.random.uniform(-1,1,(s.size,len(prev_layer))) * np.sqrt(1./s.size)).astype(np.float64)
            s.bias    = 0
            return s

        def forward(s,activated,params):
            dotted = np.dot(s.weights, activated)
            biased = dotted+s.bias
            activated = s.activation(biased)
            params.append(Callback(dotted,biased,activated))
            return activated

        def __delitem__(s, mass:int):
            s.weights = np.delete(s.weights, mass,axis=0)
            s.size    = s.size-mass
            assert s.size > 0

            s.next_layer.weights = np.delete(s.next_layer.weights, mass-1,axis=1)


            s.size = s.weights.shape[0]
            s.next_layer.size = s.next_layer.weights.shape[0]

        def __add__(s, other:int):
            assert other>0

            ws = s.weights.shape[0]
            nw = (np.random.random((other,len(s.prev_layer))) * np.sqrt(1./s.size)).astype(np.float64)
            s.weights = np.vstack([s.weights,nw])
            s.size = s.weights.shape[0]

            if s.next_layer is not None:
                nwx = (np.random.random((s.next_layer.size,other)) * np.sqrt(1. / s.size)).astype(np.float64)
                s.next_layer.weights = np.hstack([s.next_layer.weights,nwx])
                s.next_layer.size = s.next_layer.weights.shape[0]

            return s


    class LSTM(LayerUtils):
        class __LSTM:
            def __init__(s, prev_size, size,rl, l_rate):
                s.x = np.zeros(prev_size + size)
                s.prev_size = prev_size + size
                s.y = np.zeros(size)
                s.ys = size
                s.rl = rl
                s.cs = np.zeros(size)
                s.rl = size
                s.lr = l_rate
                s.r_size = r_size = (size,size+prev_size)

                s.f = np.random.random(r_size)
                s.i = np.random.random(r_size)
                s.c = np.random.random(r_size)
                s.o = np.random.random(r_size)

                s.g_f = np.zeros(r_size)
                s.g_i = np.zeros(r_size)
                s.g_c = np.zeros(r_size)
                s.g_o = np.zeros(r_size)



        def __init__(s, size,o_size,r_size,exp_o,activation=linear):
            super().__init__(size)
            s.o_size = o_size
            s.weights = ...
            s.bias    = ...
            s.prev_layer = ...
            s.activation = activation
            s.__lstm = ...
            s.r_size = r_size
            s.x = ...
            s.eo = exp_o

        def __call__(s, prev_layer):
            # rl =
            xs = prev_layer.size
            lr = 0.2
            ys = s.size
            s.input = np.zeros(s.size)
            s.i_size = xs
            s.output = np.zeros(ys)
            s.o_size = ys
            s.w_shape = (ys, ys)
            s.weights = np.random.random(s.w_shape)


            s.G = np.zeros(s.w_shape)
            s.r_size = s.r_size
            s.lr = lr

            s.gate_shape =  (s.size + 1, xs)
            s.ia     =  np.zeros(s.gate_shape)
            s.ca     =  np.zeros(s.gate_shape)
            s.oa     =  np.zeros(s.gate_shape)
            s.ha     =  np.zeros(s.gate_shape)
            s.f_gate =  np.zeros(s.gate_shape)
            s.i_gate =  np.zeros(s.gate_shape)
            s.o_gate =  np.zeros(s.gate_shape)

            s.eo = np.vstack((np.zeros(s.eo.shape[0]), s.eo.T))
            s.cell_sate = np.zeros(s.gate_shape)
            s.exp_o = np.vstack((np.zeros(s.eo.shape[0]), s.eo.T))
            # declare LSTM cell (input, output, amount of recurrence, learning rate)
            s.LSTM = Layer.LSTM.__LSTM(xs, ys, s.size, lr)


    def __class_getitem__(cls, item):
        return ...
    @staticmethod
    def using(Layer):
        return Layer


def add(x,y):
    x + y
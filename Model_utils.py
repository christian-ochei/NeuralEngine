import numpy as np
from Maths.LogicUnit import as_numpy
from random import uniform

class TrainStepToken:
    ...


class Callback:
    def __init__(s,z_dotted,z_biased,a_activated):
        s.activated = a_activated
        s.dotted    = z_dotted
        s.biased    = z_biased


class CoAdverserial():
    def __init__(s,model,x_train):
        s.generator = model
        s.x_train   = x_train

    def __and__(s, other):
        ge      = s.generator
        Network = s.generator.__class__
        model   = Network([])

        model._layers += (s.generator._layers)

        for other_ in other[:-1]:
            model._layers += (other_._layers[1:])

        # model.exponent_gradients = 5.6
        # model.exponent_gradients = 0.8
        model.exponent_gradients = 1.3
        model.randomize(0.001)

        # ge.alpha = 0.1

        d_Callback, d_output = model.feed_forward(as_numpy(s.x_train))
        d_dwb,      errors   = model.feed_backwards(other[-1], (d_Callback, d_output),)

        errors = [((e)*abs(e),x) for e,x in d_dwb]
        # d_dwb = [((e*0.01),x) for e,x in d_dwb]

        # ge.alpha = 1e-3

        ge.update_w_and_b((d_dwb[(rev:=-len(ge._layers)+1):],errors[rev:]))

        return TrainStepToken()

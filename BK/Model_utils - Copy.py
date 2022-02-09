import numpy as np
from Maths.LogicUnit import as_numpy
# from . Network import Network
from random import uniform

def prin(args):
    print(args)
    return args

class TrainStepToken:
    ...


class Callback:
    def __init__(s,z_dotted,z_biased,a_activated):
        s.dotted = z_dotted
        s.biased = z_biased
        s.activated = a_activated



class CoAdverserial():
    def __init__(s,model,x_train):
        s.generator = model
        s.x_train = x_train

    def __and__(s, other):
        ge = s.generator
        Network = s.generator.__class__
        
        model = Network([
        ])

        model._layers += (s.generator._layers)

        for other_ in other[:-1]:
            model._layers += (other_._layers[1:])
        # print('---',model[as_numpy(s.x_train)])
        d_Callback, d_output = model.feed_forward(as_numpy(s.x_train))
        d_dwb, errors = model.feed_backwards(1, (d_Callback, d_output))

        model.update_w_and_b()

        ge.update_w_and_b((reversed([x for x in reversed(d_dwb)][:len(ge._layers)-1]),None),div=114)

        return TrainStepToken()

        a_out = other[-1]
        d_models = other[:-1]

        params,output = ge.feed_forward(s.x_train)

        d_Callback,d_output = d_models[0].feed_forward(output)
        d_dwb,errors        = d_models[0].feed_backwards(100,(d_Callback,d_output))

        error = errors[-1]

        y_train = as_numpy([x for x in range(5)])

        d_wb = []
        m = output.shape[0]
        last = len(params)-1
        # pr = (2*(out-y_train)/m)

        prod = lambda:(np.dot(d_models[0]._layers[1].weights.T, error))*1e-9

        # prod = lambda:4*(np.array([x for x in range(5)]).astype('float64')-output)/m
        # prod = lambda: (np.random.uniform(-1,1,output.shape)*(d_output-a_out))/10

        errors = []

        for x in range(len(ge._layers)-1):
            # print('---')
            error = prod() * ge._layers[last-x].activation(params[last-x].activated,derivative=True)
            error = error*(abs(error)*1.3)
            errors.append(error)
            outer = (np.outer(error, params[last - 1 - x].activated))
            d_wb.append(
                (outer*np.random.uniform(-0.008,2,outer.shape),
                 np.sum(error) / m*uniform(-0.001,2)
                 )
            )
            prod = lambda:np.dot(ge._layers[last-x+1].weights.T,error)

        ge.update_w_and_b((d_wb,errors))

        return TrainStepToken()


from . import Network
class Gradients:
    def __init__(s,*args):
        s.models = args

        # super(Network, s).__init__()
        # s.__layers = Network.__layers

    def __enter__(s):
        for model in s.models:
            model.gradient_tape = True
            model.__call__ = s.__call

        return s.models

    def __exit__(s, exc_type, exc_val, exc_tb):
        # del s.__call__
        for model in s.models:
            model.gradient_tape = False
            model.__param_buffer = False




    def __call(s, loss):
        return ...

#
# class GradientSolver(Network):
#     def __init__(self,n):
#         super(GradientSolver, self).__init__(n)


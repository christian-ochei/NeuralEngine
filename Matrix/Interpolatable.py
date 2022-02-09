import numpy as np
from   random import uniform

class InterpolatableF:

    def __init__(s,n_inputs,n_outputs,min,max,spacing,func=np.zeros):
        s.n_inputs = n_inputs
        s.spacing  = spacing

        s.min  = min
        s.max  = max
        s.size = (max-min)//spacing
        s.data = np.random.uniform(-0,0,(*[s.size for _ in range(n_inputs)],n_outputs))

    def __getitem__(s, item):
        item = np.array(item)
        v = ((item-s.min)/s.spacing)
        d = v%1

        n = [slice(int(a),int(a)+2) for a in v]

        dti = np.copy(s.data[(*v.astype(int),)])
        dt  = s.data[(*n,)]

        for x in range(s.n_inputs):
            var = [0 for x in range(s.n_inputs)]
            var[x] = 1
            dti += (dt[tuple(var)]-dti)*d[x]

        return dti

    def __setitem__(s, item, value):
        value = np.array(value)
        item  = np.array(item)
        v = ((item - s.min) / s.spacing)
        d = v % 1

        diff = s.__getitem__(item)-value

        for x in range(s.n_inputs):
            var = [0 for x in range(s.n_inputs)]
            var[x] = 1

            s.data[(*(v+var).astype(int),)] -=  diff*np.mean(d)/s.n_inputs
            s.data[(*v.astype(int),)]       -=  diff*np.mean(1-d)/s.n_inputs


    def __call__(s, x):
        return s.__getitem__(x)

    def __iter__(s):
        return iter(range(s.min,s.max-s.spacing,s.spacing))


class Point:
    count = 1
    def __init__(s):
        s.x = uniform(0,5000)
        s.y = uniform(0,5000)
        Point.count += 1
        s.v = Point.count

    def __repr__(s):
        return f"<Point {s.v}>"

    def __iter__(s):
        return iter(s.inputs())


    def reset(s):
        s.x = uniform(-100, 100)
        s.y = uniform(-100, 100)

    def inputs(s):
        return np.array([s.x,s.y]).astype(float)

#
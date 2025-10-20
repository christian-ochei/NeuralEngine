# NeuralEngine

#### NeuralEngine is a small Deep learning framework I built for fast Neural Network creation and evaluation. This framework is built around numpy in order to aid fast CPU compute

#

#### The design of this framework allows easy notations, allowing the user to create, train an evaluate models with ease.

```python

from neuralengine import Layer
import neuralengine
import numpy

model = neuralengine.Network([
    Layer.Input(40),
    Layer.Dense(42,activation=neuralengine.sigmoid),
    Layer.Dense(100),
    Layer.Dense(20),
    Layer.Dense(2)])

model[numpy.random.randn(40)] = numpy.ones(2)
# equivalent square(model(numpy.random.randn(40)) - numpy.ones(40)).mean().backward().step()

```

Neural also has a simple and elegant notation for Adverserial Networks

```python
with Adverserial(gen):
    # G(X) must be tuned for D(G(X)) == 1
    gen[np.random.uniform(-1,1,5)] &= des,1

des[gen[np.random.uniform(-1, 1, 5)]] = 0
    
```

###


#

### Give it a try by running main.py

```
python main.py
```



### Contributions

#### This project is very much open contributions. Feel free to report bugs.

#

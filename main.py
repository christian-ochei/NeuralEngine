import nuralengine
import numpy
import torch
from nuralengine import Layer


model = nuralengine.Network([
    Layer.Input(40),
    Layer.Dense(42,activation=nuralengine.sigmoid),
    Layer.Dense(100),
    Layer.Dense(20),
    Layer.Dense(2)])


model.randomize_weights(0.2)


# Supports Automatic conversion from different libraries

# Run forward pass
numpy_z = model(numpy.random.uniform(0,20,40))
torch_z = model(torch.randn(40))

print(f"""
    {numpy_z=}
    {torch_z=}
""")


# Train step is as simple as an assign comprehension

for _ in range(100):
    # Fits model Based on an assign operation

    # equivalent to model(input)
    model[numpy.random.randn(40)] = numpy.ones(2) #
    # equivalent square(model(numpy.random.randn(40)) - numpy.ones(40)).mean().backward().step()


with model.adv
print(model[numpy.random.randn(40)])


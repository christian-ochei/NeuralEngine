from nuralengine import Network,Layer,MSE,sigmoid

# From my Pygame GitHub page
from Pygame import *
import numpy as np

y_train = [0.8,0.2,1,0.3,0.4]
model = Network([
    Layer.Input(10),
    Layer.Dense(20,sigmoid),
    Layer.Dense(50,sigmoid),
    Layer.Dense(100,sigmoid),
    Layer.Dense(5),
])



import math

def GetSlopeDeg(slope):
  radians = math.atan(slope)
  return radians

def slope(c,p):
    pygame.draw.line(screen, WHITE, (p[0],p[1]), (p[0]+100, p[1]+(100/c)), 2)
    pygame.draw.line(screen, WHITE, (p[0],p[1]), (p[0]-100, p[1]-(100/c)), 2)
    # pygame.draw.line(screen, (255, 255, 255), (0+p[0],1000+p[1]), (((2000)/c)+p[0], p[1]), 2)


c = 0.00000000001
def slope2point(s):
    return [c, (c/s)]


@Threaded
def t():
    for x in range(1000000):
        model[np.random.random(10)] = y_train
#
# @Threaded
# def run():
#     # for x in range(200):
#     #     if x%1000 == 1:

print(y_train)
print(model[np.random.random(10)])
print()
print(MSE(y_train, model[np.random.random(10)]) * 100)
@Pygame
def frame_ev():
    steps = 0

    pos = [400,400]
    for x in model.d_wb_buffer:

        #
        # # for v in range(l)
        s = x[0][0][0][0]
        # steps += 10/s*20
        # slope(s,(400-steps,400+steps*2))

        pos[0] += slope2point(2)[0]
        pos[1] += slope2point(2)[1]

        dot(*pos)






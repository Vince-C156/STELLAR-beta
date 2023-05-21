import numpy as np
import matplotlib.pyplot as plt
from dynamics import chaser_discrete
import os
from visualizer_all import render_visual
from time import sleep
os.chdir('runs')


typegraph = input('(vbar{n}): ')
path = os.getcwd()
data_dir = os.path.join(path, typegraph)
num = len(os.listdir(data_dir))

if num > 0:
    num -= 1

#data_file_name = f'chaser{num}.txt'

only_x0 = str(input('Render only inital States [Y] [N]: '))

if only_x0 == 'Y':
    vis_obj = render_visual(typegraph, True)
else:
    vis_obj = render_visual(typegraph)
vis_obj.render_animation()
#vis_obj.save()

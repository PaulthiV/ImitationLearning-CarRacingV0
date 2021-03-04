import gym
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time

env_name = "CarRacing-v0"
env = gym.make(env_name)


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    isopen = True
    action = list()
    c = 0

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()

            ## To save state as rgb array: 
            """v = env.render(mode='rgb_array')
            filename = "testing/test/%s_pics.png" % c
            c += 1
            img = Image.fromarray(v, 'RGB')
            img.save(filename)
            v.tofile(filename,sep=',')"""

            ## To view the output:
            """print("shape of s:", s.shape, "actions:",a, type(a))
            rgb_weights = [0.2989, 0.5870, 0.1140]
            gs_im = np.dot(v[...,:3], rgb_weights)
            print(gs_im.shape)
            plt.imshow(v)
            plt.show()
            break"""
            
            ## to save the action:
            """action.append(a.tolist())
            filename = "data/%s_data.csv" % c
            df = pd.DataFrame(action)
            df.to_csv("testing/action.csv", index=False)"""


            if done or restart or isopen == False:
                break
       
    env.close()
    

    
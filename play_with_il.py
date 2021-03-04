import numpy as np
import keras
from keras.models import Model
import gym
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import cv2

env_name = "CarRacing-v0"
env = gym.make(env_name)

def data_processing(v_data):
    h = 350
    w = 600
    rh = int(h/2)
    rw = int(w/2)
    test_data = []
    filename = 'image.png'

    img = Image.fromarray(v_data, 'RGB')
    img.save(filename)
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    crp_array = img_array[0:h, 0:w]
    new_array = cv2.resize(crp_array, ((rw,rh)))
    new_array = new_array.T
    new_array = new_array.reshape(-1, rh, rw, 1)
    new_array = np.array(new_array)
    new_array = new_array / 255.0
    test_data.append(new_array)

    pred_a = model_pred(test_data)

    return pred_a

def model_pred(test_data):
    keras_model = keras.models.load_model('models/final/model_keras_trial.model')
    keras_predictions = []
    prediction = keras_model.predict(test_data)

    a = np.argmax(prediction[0])
    b = np.argmax(prediction[1])
    c = np.argmax(prediction[2])

    keras_predictions.append([a,b,c])

    pred_df = pd.DataFrame(keras_predictions)
    pred_dir = pd.DataFrame(pred_df[0])
    pred_dir = pred_dir.replace(0, -1.0)
    pred_dir = pred_dir.replace(1, 0.0)
    pred_dir = pred_dir.replace(2, 1.0)

    pred_acc = pd.DataFrame(pred_df[1])
    pred_acc = pred_acc.replace(0, 0.0)
    pred_acc = pred_acc.replace(1, 1.0)

    pred_dec = pd.DataFrame(pred_df[0])
    pred_dec = pred_dir.replace(0, 0.0)
    pred_dec = pred_dir.replace(1, 0.8)

    pred_a = [pred_dec.values[0][0], pred_acc.values[0][0], pred_dec.values[0][0]]

    return pred_a



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
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            v = env.render(mode='rgb_array')
            data_array = data_processing(v)

            s, r, done, info = env.step(np.array(data_array))
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()

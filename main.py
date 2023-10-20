from camera import webcam
import threading
from tqdm import tqdm
from network import build, architectures
import matplotlib.pyplot as plt
import pickle
import os
from utils import *
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def foreground(cam):
    while True:
        print("foreground active")
        cam.deactivate()
        break


def turn_on():
    cam = webcam.WebCam()
    webcam_function = threading.Thread(name='background', target=cam.activate)
    network = threading.Thread(name='foreground', target=foreground)
    webcam_function.start()
    network.start()


if __name__ == "__main__":
    [x_train, y_train], [x_test, y_test] = build.dataset()
    build.model([x_train, y_train], [x_test, y_test])

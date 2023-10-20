import os
import yaml


def get_config():
    with open("config.yml", "rb") as file:
        contents = yaml.safe_load(file)
    return contents

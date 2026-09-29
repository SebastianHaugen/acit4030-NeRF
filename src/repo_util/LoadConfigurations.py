#Imports
import yaml

#Loading config yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#Retriving values
preview = config["preview"]["the-preview"]

print(preview)

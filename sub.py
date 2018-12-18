
import subprocess, time, os, sys

input_dir = '/workspace/images/'

out = ""

# p = subprocess.Popen([sys.executable,
#                      "python", "/workspace/rude-carnie/guess.py",
#                      "--model_type", "inception",
#                      "--model_dir", "/workspace/22801",
#                      "--filename", "/workspace/images/1000_16_M_60+_0.0543072_1.86479.jpg"],
#                      stdout=subprocess.PIPE,
#                      stderr=subprocess.STDOUT)

p = subprocess.Popen([sys.executable,
                     "/workspace/rude-carnie/guess.py",
                     "--model_type", "inception",
                     "--model_dir", "/workspace/22801",
                     "--filename", "/workspace/images/1000_16_M_60+_0.0543072_1.86479.jpg"],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)

with open("out.log", "w") as f:
    f.write(p.stdout.read())
# print(p.stdout.read())
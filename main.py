from train import *
from predict import *
import os
print("You are executing dog_classifier")

while True:
    print("Type 't' for train, 'p' for prediction, and 'q' for quit")
    input_1=input()
    if input_1=='t':
        train()
    if input_1=='p':
        print("What is your file?")
        input_2=str(input())
        if os.path.isfile(input_2):
            predict(input_2)
    if input_1=='q':
        break


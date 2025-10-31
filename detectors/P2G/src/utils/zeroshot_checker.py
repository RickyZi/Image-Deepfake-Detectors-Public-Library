import pickle
import os
import glob

dataroot = "/media/mmlab/Volume2/TrueFake"

if __name__ == "__main__":
    with open("./classes.pkl", "rb") as f:
        results_new = pickle.load(f)
    with open("./classes_old.pkl", "rb") as f:
        results_old = pickle.load(f)
    
    for key, value in list(results_new.items())[:10]:
        print(key, value)
    for key, value in list(results_old.items())[:10]:
        print(key, value)

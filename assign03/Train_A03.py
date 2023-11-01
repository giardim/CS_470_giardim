import tensorflow as tf
import os

def train_A03():
    ds = tf.data.Dataset("BCCD")

    if (os.path.exists("assign03/output_wbc")):
        overwrite = input(f"***WARNING*** 'output_wbc' already exists. Running this program will overwrite the existing file.
                           Continue? (Y\n)")
        overwrite = overwrite.toUpperCase()
        while (overwrite != "Y" or overwrite != "N"):
            overwrite = input(f"***WARNING*** {overwrite} is not a valid input. Please enter 'Y' or 'n'")
        if (overwrite == "N"):
            print("***Ending program***")
            return
    file = open('assign03/output_wbc/RESULTS_WBC.txt')
    
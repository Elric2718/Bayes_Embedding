import tensorflow as tf
import numpy as np
import input

FLAGS = {}
FLAGS["data_proc_sh"] = "./data_proc.sh"
FLAGS["file_dir"] = "../data"
FLAGS["input_name"] = "test.txt"
FLAGS["output_name"] = "test"

train, test = input.SplitData(FLAGS, None, None, 0.8, 5)

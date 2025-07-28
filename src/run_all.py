import os

models = ["gru", "lstm", "convgru"]
versions = ["unfiltered", "6class", "5class"]

for model in models:
    for version in versions:
        cmd_train = f'python src/train_model.py --arch {model} --data_version {version} --batch_size 256 --fp16'
        print("Running:", cmd_train)
        os.system(cmd_train)

for model in models:
    for version in versions:
        cmd_eval = f'python src/eval_model.py --arch {model} --data_version {version} --batch_size 256'
        print("Running:", cmd_eval)
        os.system(cmd_eval)

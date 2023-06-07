import os
import shutil
folders = ['test', 'test2', 'train', 'train2', '0_0_train-test_move.py']

for filename in sorted(os.listdir(os.getcwd())):
    if filename not in folders:
        lines = open(filename, 'r').readlines()
        print(filename, lines[0][-4:-2])
        if '}' in lines[0][-4:-2]:
            shutil.move(filename, './test')

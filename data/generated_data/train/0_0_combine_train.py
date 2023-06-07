import os

# combine all of the files into 1
master = open('./0_0_master_train.txt', 'w')
for filename in os.listdir(os.getcwd()):
    if filename != '0_0_master_train.txt' and filename != '0_0_combine_train.py' and filename != '0_0_master_train_clean.txt':
        master.write("{}\n".format(open(filename, 'r').read()))
print("completed compilation")

# get rid of any blank lines and get rid of numbers in the begining of file
master_lines = open('./0_0_master_train.txt', 'r').readlines()
master_clean = open('./0_0_master_train_clean.txt', 'w')
for line in master_lines:
    # write line if it isn't blank
    if line != "" and line != '\n':
        # get the index of the first '.'
        pos = 0
        for c in range(len(line)):
            if line[c] == '.':
                pos = c
                break
        
        # check if there are 3 '|'. This is a common error. Need to replace the middle one with a ' '
        bar_list = []
        for c in range(len(line)):
            if line[c] == '|':
                bar_list.append(c)
        
        if len(bar_list) == 3:
            line = line[:bar_list[1]] + ' ' + line[bar_list[1]+1:]

        # if the line is going to start with a space, chop the text further.
        if line[pos+1] == ' ':
            master_clean.write("{}".format(line[pos+2:]))
        else:
            master_clean.write("{}".format(line[pos+1:]))

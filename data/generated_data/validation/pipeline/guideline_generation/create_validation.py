import torch
import time
import datetime
import re
import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration


def main():
    max_length = 512

    print('loading model...')
    guideliner_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    guideliner = T5ForConditionalGeneration.from_pretrained("TrevorAshby/guideliner")

    device_id = '0'
    device = torch.device('cuda:'+device_id)

    guideliner.to(device)
    guideliner.eval()

    log = open('./val_out.txt', 'w')

    # grab the input line by line
    file = open('../../../train/0_0_master_train_clean.txt', 'r')
    lines = file.readlines()

    the_id = 0
    # loop through to get 100 examples
    for i in range(100):
        if i % 5 == 0:
            print('{}/100'.format(i))
        rand = random.randrange(len(lines))
        line = lines[rand]
        #print("line: ", line)
        history, topics, gl = line.split('|')

        guide_in_str = history + '| ' + topics

        in_ids = guideliner_tokenizer(guide_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        guideline_example = guideliner.generate(in_ids.to(device), max_new_tokens=50)
        guideline = guideliner_tokenizer.decode(guideline_example[0], skip_special_tokens=True)

        # write ID, HISTORY, OUR_OUTPUT, and BASELINE_OUTPUT line to file
        log.write('{}\t{}\t{}\n'.format(the_id, history, guideline))
        the_id += 1

if __name__ == "__main__":
    main()


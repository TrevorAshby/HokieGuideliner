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
    topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
    topic_detector = AutoModelForSeq2SeqLM.from_pretrained('TrevorAshby/topic-detector')

    device_id = '0'
    device = torch.device('cuda:'+device_id)

    topic_detector.to(device)
    topic_detector.eval()

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
        history, topics, guideline = line.split('|')

        # history -> topic_modified_in
        hist2 = re.split('[A,B]: ', history)
        hist2 = hist2[1:] # need this to get rid of '' that was extracted from split
        #print(hist2)

        topic_modified_in = ''
        for i, turn in enumerate(hist2):
            if i % 2 == 0:
                spot = 'Robot: '
            else:
                spot = 'Human: '
            topic_modified_in += spot + turn + ' '
            if (i < len(hist2)-1) and (i+1 % 2 == 0):
                topic_modified_in += '[ENDOFTURN]'

        topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + topic_modified_in + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
        # print(topic_in_str)
        user_input_ids = topic_tokenizer(topic_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        topic_pref_example = topic_detector.generate(user_input_ids.to(device), max_new_tokens=128)
        topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)

        # write ID, HISTORY, OUR_OUTPUT, and BASELINE_OUTPUT line to file
        log.write('{}\t{}\t{}\n'.format(the_id, history, topic_pref))
        the_id += 1

if __name__ == "__main__":
    main()


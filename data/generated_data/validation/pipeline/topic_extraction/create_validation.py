import torch
import time
import datetime
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration


def main():
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
        rand = random.randrange(len(lines))
        line = lines[rand]
        history, topics = line.split('|')

        # history -> topic_modified_in
        hist2 = re.split('.:', history)

        print(hist2)

        topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + topic_modified_in + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
        user_input_ids = topic_tokenizer(topic_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        topic_pref_example = topic_detector.generate(user_input_ids.to(device), max_new_tokens=128)
        topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)

        # write ID, HISTORY, OUR_OUTPUT, and BASELINE_OUTPUT line to file
        log.write('{}\t{}\t{}\t{}\n'.format(the_id, history, topic_pref, comp_response))
        the_id += 1


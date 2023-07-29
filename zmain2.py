import torch
import time
import datetime
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

def main():
    START_MSG = "Hello! I am Hokiebot. I love to have conversations. Say something to me..."
    CONV_HISTORY = [(START_MSG, '')]
    MAX_CONV_TURNS = 3 # set to 0 if you just want user input passed through.

    USE_BLENDERBOT = True
    USE_INSTRUCTDIALOGUE = False

    COUT_LOGGING = False

    device_id = '0'
    device = torch.device('cuda:'+device_id)
    log = open('./main_log_zmain2.txt', 'a+')

    if USE_BLENDERBOT:
        generator_used = 'BLENDERBOT'
    elif USE_INSTRUCTDIALOGUE:
        generator_used = 'INSTRUCTDIALOGUE'
    log.write(generator_used + ' - ' + 'PREV_TURNS:{}'.format(MAX_CONV_TURNS) + ' - ' + str(datetime.datetime.now()) + '\n')
    max_length = 512


    # load the models
    print('**** LOADING MODELS ****')
    topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
    topic_detector = AutoModelForSeq2SeqLM.from_pretrained('TrevorAshby/topic-detector')

    guideliner_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    guideliner = T5ForConditionalGeneration.from_pretrained("TrevorAshby/guideliner")

    blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    blen_model = AutoModelForSeq2SeqLM.from_pretrained("TrevorAshby/blenderbot-400M-distill")

    # blen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # blen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    # inst_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
    # inst_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")

    # load the states & set to eval
    print('**** LOADING CHECKPOINTS & TO GPU****')
    # topic_detector.load_state_dict(torch.load('./model/DIAL-BART0_9_checkpoint.pt'))
    # guideliner.load_state_dict(torch.load('./model/guideliner.pt'))
    # blen_model.load_state_dict(torch.load('./model/blenderbot.pt'))
    # inst_model.load_state_dict(torch.load('./model/intructdialogue_V4.pt'))

    topic_detector.to(device)
    guideliner.to(device)
    blen_model.to(device)
    # inst_model.to(device)

    guideliner.eval()
    blen_model.eval()
    # inst_model.eval()

    print("========== Welcome to the Hokiebot Topic based generator =========")
    print("==========      If you wish to exit, type \'goodbye\'      =========")
    print("Hokiebot: {}".format(START_MSG))
    log.write("Hokiebot: {}\n".format(START_MSG))
    while(1):
        # take in user response
        user_response = input("You: ")
        log.write("You: {}\n".format(user_response))

        # exit if the user says goodbye
        if user_response == 'goodbye':
            break

        full_time = time.time()

        # get the topic preferences
        topic_time = time.time()
        topic_modified_in = ''
        if MAX_CONV_TURNS != 0:
            for i, turn in enumerate(CONV_HISTORY):
                if turn[1] != '':
                    topic_modified_in += 'Human: ' + turn[0] + ' '
                else:
                    topic_modified_in += 'Robot: ' + turn[0] + ' '
                if (i < len(CONV_HISTORY)-1) and (i+1 % 2 == 0):
                    topic_modified_in += '[ENDOFTURN]'
        # print(topic_modified_in)
        topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + topic_modified_in + 'Human: ' + user_response + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
        user_input_ids = topic_tokenizer(topic_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        topic_pref_example = topic_detector.generate(user_input_ids.to(device), max_new_tokens=128)
        topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)
        topic_time = (time.time() - topic_time)%60

        if COUT_LOGGING:
            print("=== User Input: {} ===".format(user_response))
            print("=== Topic Pref: {} ===".format(topic_pref))

        log.write("=== User Input: {} ===\n".format(user_response))
        log.write("=== Topic Pref: {} ===\n".format(topic_pref))
        # add to the conversation history, if at max, remove 1
        if len(CONV_HISTORY) == MAX_CONV_TURNS:
            CONV_HISTORY.pop(0)
        CONV_HISTORY.append((user_response, topic_pref))

        # generate the guideline with input and topic
        guideline_time = time.time()
        #! modify conversation history so that it makes sense -> A: B: A: B:|{{}}
        guide_in_str = ''
        topics_combi = ''
        if MAX_CONV_TURNS != 0:
            for i, turn in enumerate(CONV_HISTORY):
                # B's
                if i+1 % 2 == 0:
                    guide_in_str += 'B: ' + turn[0]
                else:
                    guide_in_str += 'A: ' + turn[0]
                topics_combi += turn[1]
            
            topics_combi += turn[-1] + ' ' + turn[-2]
            guide_in_str += '| ' + topics_combi
        else:
            guide_in_str += 'A: {}| {}'.format(user_response, topic_pref)

        in_ids = guideliner_tokenizer(guide_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        guideline_example = guideliner.generate(in_ids.to(device), max_new_tokens=50)
        guideline = guideliner_tokenizer.decode(guideline_example[0], skip_special_tokens=True)
        guideline_time = (time.time() - guideline_time)%60

        # using the guideline generate using one of the models
        generated_response = ''
        generator_time = time.time()
        if USE_BLENDERBOT:
            #! modify conversation history so that it makes sense -> *chat*</s> <s>*chat*</s> <s>*chat* [GUIDELINE] *guideline text*
            blend_in_str = ''
            
            if MAX_CONV_TURNS != 0:
                for i, turn in enumerate(CONV_HISTORY):
                    blend_in_str += turn[0]
                    if (i < len(CONV_HISTORY)-1):
                        blend_in_str += '</s> <s>'
            else:
                blend_in_str = user_response

            blend_in_str += ' [GUIDELINE] ' + guideline
            blend_in_ids = blen_tokenizer([blend_in_str], max_length=128, return_tensors='pt', truncation=True)
            blend_example = blen_model.generate(**blend_in_ids.to(device), max_length=60)
            blend_response = blen_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0]
            generated_response = blend_response

        # elif USE_INSTRUCTDIALOGUE:
        #     #! modify conversation history so that it makes sense -> Robot: *chat* Human: *Chat* [ENDOFTURN] Robot:... Human:...
        #     if CONV_HISTORY[-1][1] != '':
                
        #         #topic_dict = json.loads(CONV_HISTORY[-1][1])
        #         topic_dict = re.search(r'{\"high-level\": {\"topic\": \"(.*)\", \"if', CONV_HISTORY[-1][1]).group(0).split(':')[-1].split(',')[0]
        #         if COUT_LOGGING:
        #             print(CONV_HISTORY[-1][1])
        #             print(topic_dict)
        #     else:
                
        #         #topic_dict = json.loads(CONV_HISTORY[-2][1])
        #         topic_dict = re.search(r'{\"high-level\": {\"topic\": \"(.*)\",', CONV_HISTORY[-2][1]).group(0).split(':')[-1].split(',')[0]
        #         if COUT_LOGGING:
        #             print(CONV_HISTORY[-2][1])
        #             print(topic_dict)
            
        #     topic = 'These two people are talking about ' + topic_dict
        #     #topic = "These two people are talking about books."
        #     question = "Given this conversation provided, a response following the topic guideline is"

        #     inst_modified_in = ''
        #     for i, turn in enumerate(CONV_HISTORY):
        #         if turn[1] != '':
        #             inst_modified_in += 'Human: ' + turn[0] + ' '
        #         else:
        #             inst_modified_in += 'Robot: ' + turn[0] + ' '
        #         if (i+1 % 2 == 0): # (i < len(CONV_HISTORY)-1) and 
        #             inst_modified_in += '[ENDOFTURN]'

        #     inst_in_str = "Instruction: Generate a response that following the given topic guideline.\n Input: [TOPICS] " + topic + " [CONTEXT] " + inst_modified_in + " [ENDOFDIALOGUE] [QUESTION] " + question + " [GUIDELINE] " + guideline
            
        #     if COUT_LOGGING:
        #         print('INST_IN_STR: ', inst_in_str)
            
        #     log.write('=== INST_IN_STR: {} ===\n'.format(inst_in_str))

        #     inst_in_ids = inst_tokenizer(inst_in_str, max_length=128, return_tensors='pt', truncation=True)
        #     inst_example = inst_model.generate(**inst_in_ids.to(device), max_length=60)
        #     inst_response = inst_tokenizer.batch_decode(inst_example, skip_special_tokens=True)[0]
        #     generated_response = inst_response
        #     generated_response = generated_response.replace('Robot: ', '')
        #     generated_response = generated_response.replace('Human: ', '')

        generator_time = (time.time() - generator_time)%60
        
        # add to the conversation history, if at max, remove 1
        if len(CONV_HISTORY) == MAX_CONV_TURNS:
            CONV_HISTORY.pop(0)
        CONV_HISTORY.append((generated_response, ''))

        print('Hokiebot: {}'.format(generated_response))
        log.write('Hokiebot: {}\n'.format(generated_response))
        if COUT_LOGGING:
            print('=== TIMING [tpc:{}, gdl:{}, gen:{}, tot:{}] ==='.format(topic_time, guideline_time, generator_time, (time.time() - full_time)%60))
            print('=== CONV_HISTORY [{}] ==='.format(str(CONV_HISTORY)))
        log.write('=== TIMING [tpc:{}, gdl:{}, gen:{}, tot:{}] ===\n'.format(topic_time, guideline_time, generator_time, (time.time() - full_time)%60))
        log.write('=== CONV_HISTORY [{}] ===\n'.format(str(CONV_HISTORY)))

    print('Hokiebot: Goodbye! Thanks for chatting!')
    log.write('Hokiebot: Goodbye! Thanks for chatting!\n\n')
    return 0

if __name__ == '__main__':
    main()
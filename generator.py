import json
import torch
import datetime
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
# import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import AdamW
from tqdm import tqdm  # for our progress bar
from torch.nn.parallel import DistributedDataParallel as DDP

def train(model1, model2, dl, tokenizer1, tokenizer2, num_epochs=10):
    model1.train()
    model2.train()
    print('Starting model training...')

    log = open('./logV2.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))

    criteria2 = AdamW(model2.parameters(), lr=1e-6)
    criteria1 = AdamW(model1.parameters(), lr=1e-6)

    for epoch in range(num_epochs):
        eploss2 = 0
        eploss1 = 0

        # setup loop with TQDM and dataloader
        loop = tqdm(dl, leave=True)
        for batch in loop:
            bio, boo, iio, ioo = batch #! FOR DETAIL ON THESE SEE LINE 87
            # initialize calculated gradients (from prev step)
            criteria2.zero_grad()
            criteria1.zero_grad()
            # pull all tensor batches required for training

            input_ids2 = iio.to(device)
            input_ids1 = bio.to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels_ids2 = ioo.to(device)
            labels_ids1 = boo.to(device)
            # process
            outputs2 = model2(
                input_ids=input_ids2.squeeze(1),
                labels=labels_ids2.squeeze(1),
                # attention_mask=attention_mask
            )
            outputs1 = model1(
                input_ids=input_ids1.squeeze(1),
                labels=labels_ids1.squeeze(1),
            )
            # extract loss
            loss2 = outputs2.loss
            loss1 = outputs1.loss
            # # calculate loss for every parameter that needs grad update
            loss2 = loss2.mean()
            loss1 = loss1.mean()
            
            loss2.backward()
            loss1.backward()
            # update parameters
            criteria2.step()
            criteria1.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss2=loss2.item(), loss1=loss1.item())

            eploss2 += loss2.item()
            eploss1 += loss1.item()

        log.write("Epoch:{}, EpLoss1:{}, EpLoss2:{}\n".format(epoch, eploss1/len(dl), eploss2/len(dl)))



device_id = '0' # need to change this to 6 when I am training w/ jingyuan's GPU
max_length = 128
batch_size = 5
epoch_num = 50


# A dataset that pulls from both datasets
class CombiDataset(torch.utils.data.Dataset):
    def __init__(self, blen_path, inst_path, blen_tokenizer, inst_tokenizer, max_length=50):
        self.blen_tokenizer = blen_tokenizer
        self.inst_tokenizer = inst_tokenizer
        self.blen_examples = pd.read_csv(blen_path, sep='|')
        self.inst_examples = pd.read_csv(inst_path, sep='|')
        self.max_len = max_length

    def __len__(self):
        return len(self.blen_examples)

    def __getitem__(self, idx):
        #print('===========IDX:{}==========='.format(idx))
        blen_in, blen_out = self.blen_examples.iloc[idx]
        inst_in, inst_out = self.inst_examples.iloc[idx]
        #print('blen_in:', blen_in)
        #print('blen_out:', blen_out)
        instruction = "Instruction: Generate a response that following the given topic guideline.\n"
        blen_in_out = self.blen_tokenizer(blen_in, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        blen_out_out = self.blen_tokenizer(blen_out, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        
        inst_in_out = self.inst_tokenizer(instruction+inst_in, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        inst_out_out = self.inst_tokenizer(inst_out, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids

        return blen_in_out, blen_out_out, inst_in_out, inst_out_out
        

if __name__ == '__main__':
    # set the device
    device = torch.device('cuda:'+device_id)

    # download the models
    blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400m-distill")
    blen_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400m-distill")

    inst_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
    inst_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")
    # inst_model = None
    # inst_tokenizer = None

    # create dataloader #! USING THE SIZE 360 BECAUSE THAT WAS THE LONGEST OF THE TWO FILES
    ds = CombiDataset('./data/generated_data/generator/blen_train/0_0_master_train_clean.txt', './data/generated_data/generator/inst_train/0_0_master_train_clean.txt', blen_tokenizer, inst_tokenizer, max_length)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    blen_model = torch.nn.DataParallel(blen_model, device_ids=[0])
    inst_model = torch.nn.DataParallel(inst_model, device_ids=[0])

    # load models to GPU
    blen_model.to(device)
    inst_model.to(device)
    
    
    train(blen_model, inst_model, dl, blen_tokenizer, inst_tokenizer, 10)

    #torch.save(blen_model.state_dict(), './model/blenderbot.pt')
    #torch.save(inst_model.state_dict(), './model/intructdialogue.pt')

    try:
        torch.save(blen_model.module.state_dict(), './model/blenderbot.pt')
    except AttributeError:
        torch.save(blen_model.state_dict(), './model/blenderbot.pt')

    try:
        torch.save(inst_model.module.state_dict(), './model/intructdialogue.pt')
    except AttributeError:
        torch.save(inst_model.state_dict(), './model/intructdialogue.pt')

    
    
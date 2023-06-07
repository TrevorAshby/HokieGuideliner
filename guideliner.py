import torch
import datetime
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

def train(model, dataloader, tokenizer, num_epochs=10):
    print("Starting model training...")
    criteria = torch.optim.Adam(model.parameters(), lr= 0.000001)
    log = open('./log.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))
    
    for epoch in range(num_epochs):
        eploss = 0

        for batch in dataloader:
            x, y = batch
            output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            out = output.loss
            out.backward()
            criteria.step()
            eploss += out.item()

        if epoch % 1 == 0:
            in_str = "B:What do you like to do? A:I love eating food. I really like new restaurants.|{\"high-level\": {\"topic\": \"food\", \"if_interest\": \"yes\"}, \"low-level\": {\"topic\": \"new restaurant\", \"if_interest\": \"yes\"}}"
            in_ids = tokenizer(in_str, return_tensors='pt').input_ids
            example = model.generate(in_ids.cuda(), max_new_tokens=50)
            dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
            log.write("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"\n".format(epoch, eploss/len(dataloader), in_str, dec_out))
            print("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"".format(epoch, eploss/len(dataloader), in_str, dec_out))
    
    log.close()

class GuidelineDataset(Dataset):
    def __init__(self, guideline_file, tokenizer):
        self.tokenizer = tokenizer
        self.examples = pd.read_csv(guideline_file, sep='|')

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        history, pref, guideline = self.examples.iloc[idx]
        #print("the history is: ", history)
        #print("the pref is: ", pref)
        #print("the guideline is: ", guideline)
        
        pref = tokenizer(pref, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        guideline = tokenizer(guideline, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return pref, guideline

if __name__ == '__main__':
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download the models
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    # load model to GPU
    model.to(device)

    # create dataloader
    # ds = GuidelineDataset('./data/chatgpt_data.txt', tokenizer)
    ds = GuidelineDataset('./data/generated_data/train/0_0_master_train_clean.txt', tokenizer)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    # The code below this was used to test the forward pass of model.    
    # for batch in dl:
        # print(batch[0].shape)
        # print(batch[1].shape)
        # I am squeezing the 1st dimention because I kept getting (b x ? x len) in my shape, and the ? was always '1', so I just squoze it out.
        # out = model(input_ids=batch[0].squeeze(1).cuda(), labels=batch[1].squeeze(1).cuda())
        # break

    train(model, dl, tokenizer, 20)

    #TODO Write additional lines of code to save model locally.
    

from ctypes import alignment
import json
import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').pad_token_id
BOS_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').convert_tokens_to_ids('<extra_id_0>')
EOS_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').eos_token_id
class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data_folder = data_folder
        
        self.schema_path = os.path.join(self.data_folder, "flight_database.schema")
        
        with open(self.schema_path, "r") as f:
            schema_json = json.load(f)
        self.schema_json = schema_json

        self.fixed_samples = self.sample_nl_sql(2, seed=42)

        self.encoder_ids, self.encoder_masks, self.decoder_input_ids = self.process_data(data_folder, split, self.tokenizer)
        
        

        # TODO

    def sample_nl_sql(self, k, seed=None):
        if seed is not None:
            random.seed(seed)

        nl_path = os.path.join(self.data_folder, "train.nl")
        sql_path = os.path.join(self.data_folder, "train.sql")

        with open(nl_path, "r") as f:
            nl_lines = [line.strip() for line in f.readlines()]

        with open(sql_path, "r") as f:
            sql_lines = [line.strip() for line in f.readlines()]

        indices = list(range(len(nl_lines)))
        sampled_indices = random.sample(indices, k)

        samples = [(nl_lines[i], sql_lines[i]) for i in sampled_indices]

        return samples
    
    def extract_schema(self, schema_json):
        ents = schema_json["ents"]

        lines = []
        for table, cols in ents.items():
            col_list = list(cols.keys())[:4]   # limit columns
            lines.append(f"{table}: {', '.join(col_list)}")
        return " | ".join(lines[:3])  # limit tables


    def extract_joins(self, schema):
        links = schema["links"]

        join_lines = []

        for table, relations in links.items():
            for ref_table, key in relations.items():
                join_lines.append(
                    f"{table}.{key} = {ref_table}.{key}"
                )

        return "\n".join(join_lines)
    
    def build_prompt(self, request):
        schema_str = self.extract_schema(self.schema_json)
        joins_str = self.extract_joins(self.schema_json)
        samples = self.fixed_samples

        def format_pair(nl, sql):
            return f"English: {nl}\nSQL: {sql}"

        prompt = "Translate English to SQL: "

        #prompt += format_pair(samples[0][0], samples[0][1]) + "\n\n"
        #prompt += format_pair(samples[1][0], samples[1][1]) + "\n\n"
        #prompt += f"English: {request}\nSQL: "
        return prompt

    def process_data(self, data_folder, split, tokenizer):

        encoder_ids = []
        encoder_masks = []
        decoder_input_ids = []
        #print(self.tokenizer.special_tokens_map)

        nl_path = os.path.join(data_folder, f'{split}.nl')
        input_lines = load_lines(nl_path)

        for x in input_lines:
            x = self.build_prompt(x) + x

            enc =tokenizer(
                x,
                max_length=128,
                truncation=True,
                padding=False
            )
            encoder_ids.append(enc['input_ids'])
            encoder_masks.append(enc['attention_mask'])

        if split != "test":
            sql_path = os.path.join(data_folder, f'{split}.sql')
            target_lines = load_lines(sql_path)
            for y in target_lines:
                dec = tokenizer(
                y,
                max_length=512,
                truncation=True,
                padding=False,
            
            )              
                decoder_input_ids.append(dec['input_ids'])
        else:
            decoder_input_ids = [None] * len(encoder_ids)

        return encoder_ids, encoder_masks, decoder_input_ids
        # TODO
    
    def __len__(self):
        return len(self.encoder_ids)
        # TODO

    def __getitem__(self, idx):
        return self.encoder_ids[idx], self.encoder_masks[idx], self.decoder_input_ids[idx]
        # TODO

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    
    enc_ids_list, enc_masks_list, labels_list = zip(*batch)

    encoder_ids = pad_sequence([torch.tensor(x, dtype=torch.long) for x in enc_ids_list], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(x, dtype=torch.long) for x in enc_masks_list], batch_first=True, padding_value=0)
    

    dec_inputs_list = []
    dec_targets_list = []

    for labels in labels_list:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        labels_tensor = torch.cat((labels_tensor, torch.tensor([EOS_IDX], dtype=torch.long)))
        dec_inputs = torch.cat((torch.tensor([BOS_IDX]), labels_tensor[:-1]))
        dec_inputs_list.append(dec_inputs)
        dec_targets_list.append(labels_tensor)

    decoder_inputs = pad_sequence(dec_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(dec_targets_list, batch_first=True, padding_value=PAD_IDX)

    initial_decoder_inputs = torch.full((len(batch), ), BOS_IDX, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    # TODO

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc_ids_list, enc_masks_list, labels_list = zip(*batch)
    encoder_ids = pad_sequence([torch.tensor(x, dtype=torch.long) for x in enc_ids_list], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(x, dtype=torch.long) for x in enc_masks_list], batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.full((len(batch), ), BOS_IDX, dtype=torch.long)
    clone = torch.zeros(len(batch), 1, dtype=torch.long)

    return encoder_ids, encoder_mask, clone, clone, initial_decoder_inputs
    # TODO

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):

    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import re

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from transformers import T5TokenizerFast

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').pad_token_id
BOS_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').convert_tokens_to_ids('<extra_id_0>')
EOS_IDX = T5TokenizerFast.from_pretrained('google-t5/t5-small').eos_token_id

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not", default=True)
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=6e-3)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default = 6,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default= 2,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    args = parser.parse_args()
    #test later to make sure it works
    args.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    return args


def clean_sql(sql: str) -> str:
    sql = sql.strip()

    # remove leading junk before SELECT
    sql = re.sub(r"^[^S]*SELECT", "SELECT", sql)

    # remove trailing English fragments
    sql = re.split(r"\n|English:", sql)[0]

    # fix multiple spaces
    sql = re.sub(r"\s+", " ", sql)

    return sql.strip()

def fix_truncation(q):
    q = q.strip()
    
    parts = q.split()
    for part in parts:
        if part == 'English:' and part != parts[0]:
            parts = parts[:parts.index(part)]
            break

    if not q.startswith('SELECT'):
        parts = ['SELECT'] + parts
    q = ' '.join(parts)


    # balance parentheses
    if q.count('(') > q.count(')'):
        q += ')' * (q.count('(') - q.count(')'))

    return q

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    experiment_name = args.experiment_name

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")
        

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        
        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO
    model.eval()
    criterion = nn.CrossEntropyLoss()

    
    total_loss = 0
    total_tokens = 0
    all_preds = []


    with torch.no_grad():
        for batch in tqdm(dev_loader):
            encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs = batch
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            

            #print("Encoder input ids decoded:", args.tokenizer.batch_decode(encoder_input, skip_special_tokens=True))
            #print("Decoder target ids decoded:", args.tokenizer.batch_decode(decoder_input, skip_special_tokens=True))

            non_pad = decoder_targets != PAD_IDX

            logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            )['logits']


            
            num_tokens = torch.sum(non_pad).item()
        
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            total_loss += loss.item()*num_tokens
            total_tokens += num_tokens

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=6,
                early_stopping=True,
                repetition_penalty=1.2,
                decoder_start_token_id=BOS_IDX,
                eos_token_id=EOS_IDX,
            )

            preds = args.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            
            #preds = [fix_truncation(p) for p in preds]

            #print("Preds decoded:", preds)
            
            all_preds.extend(preds)
    #all_preds = [fix_truncation(p) for p in all_preds]


    
    save_queries_and_records(all_preds, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    
    avg_loss = total_loss / total_tokens
    num_preds = len(all_preds)
    #print(model_error_msgs)
    error_rate = sum(1 for e in model_error_msgs if e != "") / num_preds
    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            encoder_input, encoder_mask = batch[:2]
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=6,
                early_stopping=True,
                repetition_penalty=1.2,
                decoder_start_token_id=BOS_IDX,
                eos_token_id=EOS_IDX,
            )

            preds = args.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            
            #preds = [fix_truncation(p) for p in preds]
            
            all_preds.extend(preds)

    #all_preds = [fix_truncation(p) for p in all_preds]
            


    save_queries_and_records(all_preds, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    print("Finished training! Starting evaluation...")
    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()

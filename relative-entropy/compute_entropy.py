from config import DATADIR, MODELDIR
""" code by Allie Lahnala, please reach of if you have questions or suggestions: alahnala@gmail.com """
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForMaskedLM
import torch
import os
from glob import glob
from collections import defaultdict
import math
import json


def main():
    """ Load data """
    if ARGS.test_group == 'annotations':
        # the annotated data comes from the training set so that's why we pass in the train path. 
        # make sure that the LM was finetuned with the -x option to exclude the annotated posts.
        data = get_data(ARGS.test_group, ARGS.all_labels_messages_train, annotations_path=ARGS.annotation_posts)
    else:
        data = get_data(ARGS.test_group, ARGS.all_labels_messages_test)

    """ Load model """
    tokenizer = AutoTokenizer.from_pretrained(ARGS.base_lm)
    tokenizer.pad_token = tokenizer.eos_token    

    """ tokenize data """
    tokenized_data = []
    lens = []
    for message in tqdm(data.message, desc='Tokenizing data'):
        tokenized = tokenizer(f"<|endoftext|> {message} <|endoftext|>")
        tokenized_data.append(tokenized)
        lens.append(len(tokenized.input_ids))

    data['message_tokens'] = tokenized_data
    data['message_lens'] = lens
    del tokenized_data, lens


    """ get losses """
    if ARGS.lm_type == 'lm':
        data['token_losses'] = causal_model_losses(data, ARGS.checkpoint_selection, ARGS.sw_size)
    elif ARGS.lm_type == 'mlm':
        data['token_losses'] = masked_model_losses(data, ARGS.checkpoint_selection)


    """ save results """
    token_loss_column = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        token_losses = {'id':[], 'token':[], 'loss':row['token_losses']}
        token_ids = row['message_tokens'].input_ids
        for t in token_ids:
            token = tokenizer.decode(t)
            token_losses['id'].append(t)
            token_losses['token'].append(token)
        token_loss_column.append(token_losses)

    data[f"token_losses"] = token_loss_column
    cols_to_save = ['post_id', 'user_id', f"token_losses"]
    data[cols_to_save].to_json(OUTFILE)
    print("saved losses to", OUTFILE)
    return

def get_data(group, all_messages_path, annotations_path=None):
    all_messages = pd.read_csv(all_messages_path)

    if group == 'annotations':
        annotations = pd.read_csv(annotations_path)

        data = all_messages[all_messages['post_id'].isin(annotations.message_id.unique())]
        return data
    else:
        parts = group.split('_')
        risk = parts[0][0].upper() + parts[0][1:]
        if risk == 'Any':
            data = all_messages[all_messages['label'] != 'No']
        else:
            data = all_messages[all_messages['label'] == risk]

        if parts[-1] == 'all':
            return data
        elif parts[-1] == 'sw':
            return data[data['subreddit'] == 'SuicideWatch']

def masked_model_losses(data, model_checkpoint):
    print("not implemented")
    quit()



def causal_model_losses(data, model_checkpoint,sw_size):
    """TODO: change implementation to improve efficiency"""
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(ARGS.device)

    data_token_losses = [] # will have an entry per post
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Getting token losses from causal model"):
        encodings = row['message_tokens']
        seq_len = len(encodings.input_ids)

        losses = [] # should have the loss for each token
        for target_loc in range(0, seq_len):
            beg_loc = max(target_loc-sw_size, 0)
            input_ids = torch.tensor(encodings.input_ids[beg_loc:target_loc+1]).to(ARGS.device)
            target_ids = input_ids.clone()
            target_ids[:-1] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
            loss = float(outputs.loss.detach().cpu())
            losses.append(loss)
        data_token_losses.append(losses)   

    return data_token_losses   

def get_checkpoint_states(model_dir, train_group): 
    """
    Returns:
        - the path to the last trainer state (it includes all loss checkpoints)
        - list of checkpoint step values
    """
    checkpoint_path = os.path.join(model_dir, train_group, "checkpoint-*", 'trainer_state.json') 
    checkpoint_dirs = glob(checkpoint_path)
    if len(checkpoint_dirs) == 0:
        print(checkpoint_path)
        print(f"No trainer state found for {train_group}")
        return False
    
    checkpoint_step_values = [int(item.split('/')[-2].replace("checkpoint-", '')) for item in checkpoint_dirs]

    last_checkpoint = max(checkpoint_step_values)    
    trainer_state_path = checkpoint_path.replace("*", str(last_checkpoint))
    return trainer_state_path, checkpoint_step_values

def get_closest_to_min_eval_loss(df, checkpoint_steps, lms_path, lm):
    min_step = df.sort_values(by='Eval loss').index[0]
    dists = [abs(min_step-cv) for cv in checkpoint_steps]
    idx = dists.index(min(dists))
    closest_to_min_eval_step = checkpoint_steps[idx]
    checkpoint_path = os.path.join(lms_path, lm, f"checkpoint-{closest_to_min_eval_step}")
    return checkpoint_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog="python compute_entropy.py",
                    description="Gets the token entropies for the test group data from the train group's finetuned language model.")
    # input and output settings
    parser.add_argument('-tr', '--train_group', choices=['low_risk_sw', 'high_risk_sw', 'moderate_risk_sw', 'low_risk_all', 'high_risk_all', 'moderate_risk_all', 'no_risk_sw', 'no_risk_all', 'any_risk_sw', 'any_risk_all'], required=True, help="This specifies the language model that we'll use to compute losses.")
    parser.add_argument('-te', '--test_group', choices=['low_risk_sw', 'high_risk_sw', 'moderate_risk_sw', 'low_risk_all', 'high_risk_all', 'moderate_risk_all', 'no_risk_sw', 'no_risk_all', 'any_risk_sw', 'any_risk_all', 'annotations'], required=True, help="This specifies the data that the language model will be run on for losses.")    
    parser.add_argument('-d', '--model_output_dir', type=str, help="Directory where the finetuned models are saved.", default=MODELDIR)
    parser.add_argument('-o', '--token_entropies_dir', default=os.path.join(MODELDIR, '_token_entropies'), type=str, help="Directory path where you want to save the token entropies.")
    parser.add_argument('--all_labels_messages_train', default=os.path.join(DATADIR, "all_labels_messages_train.csv"), type=str, help="Path to training data file.")
    parser.add_argument('--all_labels_messages_test', default=os.path.join(DATADIR, "all_labels_messages_test.csv"), type=str, help="Path to test data file.")    
    # modeling settings
    parser.add_argument('-m', '--base_lm', choices=['distilgpt2', "lsanochkin/deberta-large-feedback", 'microsoft/deberta-base'], default='distilgpt2')
    parser.add_argument('-lm', '--lm_type', choices=['lm', "mlm"], default="lm", help="lm: CausalLM | mlm: MaskedLM. Not implemented for mlm yet.")
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('-sw', '--sw_size', default=128, type=int, help="Context window size preceding target token.")
    

    parser.add_argument('--checkpoint_selection', help='Strategy for choosing the model checkpoint to load. Use a) min_eval_loss if you want to choose based on the minimum loss on the val set during training, b) last to use the last checkpoint, or c) a path to the checkpoint directory, e.g., model_dir/train_group/checkpoint-1000. ', default='min_eval_loss')


    ARGS = parser.parse_args()

    if ARGS.checkpoint_selection == 'min_eval_loss':
        # get the checkpoint closest to the min eval loss step
        trainer_state_file, checkpoint_steps = get_checkpoint_states(ARGS.model_output_dir, ARGS.train_group)

        losses = defaultdict(lambda:{'Train loss':math.nan, 'Eval loss':math.nan, "epoch":math.nan})
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)

        for item in trainer_state['log_history']:
            losses[item['step']]['epoch'] = item['epoch']
            if 'eval_loss' in item:
                losses[item['step']]['Eval loss'] = item['eval_loss']
            if 'loss' in item:
                losses[item['step']]['Train loss'] = item['loss']
        df = pd.DataFrame(losses).T

        ARGS.checkpoint_selection = get_closest_to_min_eval_loss(df, checkpoint_steps, ARGS.model_output_dir, ARGS.train_group)
    elif ARGS.checkpoint_selection == 'last':
        trainer_state_file, checkpoint_steps = get_checkpoint_states(ARGS.model_output_dir, ARGS.train_group)
        ARGS.checkpoint_selection = trainer_state_file.replace('/trainer_state.json', '')

    try:
        assert os.path.isfile(os.path.join(ARGS.checkpoint_selection, 'model.safetensors'))
    except:
        print("Can't find model checkpoint from", ARGS.checkpoint_selection)
        quit()
    print("Using model checkpoint from:", ARGS.checkpoint_selection)
    
    OUTFILE = os.path.join(ARGS.token_entropies_dir, f"{ARGS.train_group}_on_{ARGS.test_group}.json")
    os.makedirs(ARGS.token_entropies_dir, exist_ok=True)
    print("Token entropies will be saved at:", OUTFILE)

    main()
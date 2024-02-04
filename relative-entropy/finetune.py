""" code by Allie Lahnala, please reach of if you have questions or suggestions: alahnala@gmail.com """
from config import DATADIR, MODELDIR
import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, Trainer,DataCollatorForLanguageModeling
from transformers import AutoConfig
import torch
from collections import defaultdict
import math
import os

def main():
    """ Load data """
    data = get_data(ARGS.train_group, ARGS.keep_annotations)

    """ tokenize data """
    tokenizer = AutoTokenizer.from_pretrained(ARGS.base_lm)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_data = []
    for message in tqdm(data.message, total=len(data), desc='Tokenizing'):
        tokenized = tokenizer(f"<|endoftext|> {message} <|endoftext|>", max_length=128, padding=True, truncation=True, stride=3, return_overflowing_tokens=True, return_special_tokens_mask=True, return_tensors="pt").to(ARGS.device)
        tokenized_data.append(tokenized)
    data['message_tokens'] = tokenized_data
    del tokenized_data


    """ Make Pytorch Dataset """
    raw_dataset = defaultdict(lambda:[])
    raw_dataset['index'] = []
    for idx, row in data.iterrows():
        tokens = row.message_tokens
        for input_ids, attention_mask, overflow_to_sample_mapping in zip(tokens.input_ids, tokens.attention_mask, tokens.overflow_to_sample_mapping):
            raw_dataset['index'].append(idx)
            raw_dataset['message'].append(row.message)
            raw_dataset['input_ids'].append(input_ids)
            raw_dataset['attention_mask'].append(attention_mask)
            raw_dataset['overflow_to_sample_mapping'].append(overflow_to_sample_mapping)

    # in current implementation, data is shuffled when loaded.
    train_len = int(ARGS.train_proportion*len(raw_dataset['input_ids']))
    train_encodings = {'input_ids':raw_dataset['input_ids'][:train_len], 'attention_mask':raw_dataset['attention_mask'][:train_len]}
    eval_encodings = {'input_ids':raw_dataset['input_ids'][train_len:], 'attention_mask':raw_dataset['attention_mask'][train_len:]}
    train_dataset = Dataset(train_encodings)
    eval_dataset = Dataset(eval_encodings)
    print("Train data size:", len(train_dataset), "Eval data size:", len(eval_dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    """ Train model """
    configuration = AutoConfig.from_pretrained(ARGS.base_lm)
    configuration.hidden_dropout_prob = ARGS.hidden_dropout_prob
    model = AutoModelForCausalLM.from_pretrained(ARGS.base_lm, config=configuration).to(ARGS.device)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy=ARGS.eval_strategy,
        learning_rate=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        num_train_epochs=ARGS.num_epochs,
        save_strategy=ARGS.save_strategy,
        logging_first_step=True,
        logging_dir=ARGS.logging_dir,
        logging_strategy='steps',
        logging_steps=ARGS.logging_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    print("Checkpoints saved to", checkpoint_dir)

def get_data(group, keep_annotations):
    all_messages = pd.read_csv(ARGS.all_labels_messages_train).sample(frac=1)

    if keep_annotations:
        annotated_post_ids = ['29d4au', '1ulqbn', '1y8cw4', 'ml312', '3iu651', '1end6q',
                                '34fj7p', '1811it', 'dtyiy', '3hmbre', '3i5n1a', '3j48ec',
                                '1kmzqg', 'rsk6g', '2hurih', '2nr23j', '1yc6u7', '1zif0m',
                                '2f9g97', '2nsqqm', '2swqdo', '3ghxts', '2dms1j', '2nzfj4',
                                '1vzleb', '3g31xw', '2deltm', '1el5om', '1jpm72', '2i87k0',
                                '2lsj9g', '2lxop0', '3hmw9z', '3iaxao', '27j41i', '37js2i',
                                '1j7nhr', '3eaouc', '2r2hkk', '35wrom', '2vhag4', '276h4l',
                                '3e1zc6', '2vxpos', '1wyqem', '1izo4t', '1d2gxo', '3ixiyn',
                                '17j9yv', 'd82jc'] # post ids that we have annotated internally
        all_messages = all_messages[~all_messages['post_id'].isin(annotated_post_ids)]

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


    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='finetune',
                    description='Finetune a language model on a group of CLPsych users based on risk level.')
    # input and output settings
    parser.add_argument('-g', '--train_group', choices=['low_risk_sw', 'high_risk_sw', 'moderate_risk_sw', 'low_risk_all', 'high_risk_all', 'moderate_risk_all', 'no_risk_sw', 'no_risk_all', 'any_risk_sw', 'any_risk_all'], required=True)
    parser.add_argument('-tr', '--all_labels_messages_train', default=os.path.join(DATADIR, "all_labels_messages_train.csv"), type=str, help="Path to training data file.")
    parser.add_argument('-d', '--model_output_dir', type=str, help="Directory for saving the model checkpoints. They will be saved at [args.dir]/[args.train_group]. Recommended to make this unique to your experiment.", default=MODELDIR)
    parser.add_argument('-k', '--keep_annotations', action='store_true', default=False, help='Pass in -k if you want to include the posts that we annotated internally.')
    parser.add_argument('--logging_dir', default=None, type=str, help="Path to a logging dir if you don't want to use the default.")
    # modeling params
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('-m', '--base_lm', choices=['distilgpt2'], help="Name of pre-trained language model you want to finetune.", default='distilgpt2')
    parser.add_argument('-e', '--num_epochs', default=10, type=int)
    parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float)
    parser.add_argument('-wd', '--weight_decay', default=.01, type=float)
    parser.add_argument('-ss', '--save_strategy', default="epoch", type=str)
    parser.add_argument('-es', '--eval_strategy', default="steps", type=str)
    parser.add_argument('--logging_steps', default=50, type=int, help="Number of update steps between logs")
    parser.add_argument('--hidden_dropout_prob', default=.1, type=float, help="Dropout")
    parser.add_argument('--train_proportion', default=.9, type=float, help="Proportion of data to use for train, the rest will be used for eval.")
    
    ARGS = parser.parse_args()  

    checkpoint_dir = os.path.join(ARGS.model_output_dir, ARGS.train_group)
    print(f"Finetuned model checkpoints will be saved at {checkpoint_dir}")
    os.makedirs(f'{checkpoint_dir}', exist_ok=True)

    main()
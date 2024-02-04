import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import json
from glob import glob
import difflib
from config import DATADIR, MODELDIR


def main():

    post_df = pd.read_csv(ARGS.all_messages)   
    post_df = post_df[post_df['post_id'].isin(POST_IDS)].set_index('post_id')  
    sentences_df = pd.read_csv(ARGS.sent_tokenized_file)
    sentences_df = sentences_df[sentences_df['post_id'].isin(POST_IDS)]  

    """get token entropy file paths """
    entropy_files = glob(os.path.join(ARGS.token_entropies_dir, '*.json'))
    relevant_entropy_files = []
    for fi in entropy_files:
        test_group = fi.replace('.json', '').split('_on_')[-1].replace("_all", "").replace("_sw", "")
        if "no_risk" not in test_group:
            relevant_entropy_files.append(fi)

    token_entropies_df = load_token_entropies(relevant_entropy_files, post_df)

    """ Compute token-level entropy differences """
    post_ents_and_ent_diffs = compute_entropy_differences(token_entropies_df, post_df)
    
    """ INDEX TEST DATA TOKENS INTO MESSAGE STRING POSITIONS -- this had to involve some manual tricks because of encoding differences """
    post_token_index_df = index_post_tokens(post_ents_and_ent_diffs)
    

    """ INDEX SENTENCE DATA TOKENS INTO MESSAGE STRING POSITIONS """
    sentence_token_index_df = index_sentence_tokens(sentences_df)
    post_token_index_df = pd.concat([post_token_index_df, sentence_token_index_df], axis=1)


    """ check on matches """
    matching = post_token_index_df.message_strip_du == post_token_index_df.message_strip_t
    verify_maps(post_token_index_df, matching)    


    """ add ent diffs to post_token_index_df """
    post_token_index_df = pd.concat([post_token_index_df, post_ents_and_ent_diffs], axis = 1)#.columns


    """ Map and aggregate """
    sentence_tokens_df = map_tokens_to_sentences(post_token_index_df, post_ents_and_ent_diffs, sentences_df)
    df, diff_aggregations = aggregate(sentence_tokens_df)

    outfile = os.path.join(ARGS.outdir, f'{ARGS.test_group}_sentence_entropies.json')
    print(f'SAVING {outfile}')    
    df.to_json(outfile)
    

    print("\nPrinting top 5 sents by agg differences")
    for col in diff_aggregations.columns:

        matching = df.sort_values(by=col, ascending=False)#[['sentence', 'label']]
        print(f"== {col} ==")
        for sent in matching.sentence.values[:5]:
            print(f"\t{sent}")

    print("\nSentence entropies file is saved at:", outfile)
    return

    
def load_token_entropies(entropy_files, post_df):
    """ load token losses """
    post_id2entropy_lists = defaultdict(lambda:{})
    all_keys = defaultdict(lambda:0)

    for fi in tqdm(sorted(entropy_files, reverse=True)):
        fi_name = fi.split('/')[-1]
        train_group = fi_name.replace('.json', '').split('_on_')[0]
        all_keys[train_group] += 1

        df = pd.read_json(fi)
        df = df[df['post_id'].isin(POST_IDS)]

        for _, row in df.iterrows():
            post_id2entropy_lists[row.post_id][train_group] = row.token_losses

    entropy_columns = defaultdict(lambda:[])
    sg = list(all_keys.keys())[0]
    for post_id, row in tqdm(post_df.iterrows(), total=len(post_df), desc="Getting token entropy columns for each post"):
        token_ids = post_id2entropy_lists[post_id][sg]['id'][1:]
        tokens = post_id2entropy_lists[post_id][sg]['token'][1:]
        entropy_columns['post_id'].append(post_id)
        entropy_columns['token_ids'].append(token_ids)
        entropy_columns['tokens'].append(tokens)


        entropy_dict = post_id2entropy_lists[post_id]
        for m in all_keys:
            if m in entropy_dict:
                entropy_columns[m].append(np.array(entropy_dict[m]['loss'][1:]))
            else:
                entropy_columns[m].append(None)

    entropy_columns = pd.DataFrame(entropy_columns).set_index('post_id')
    return entropy_columns


def compute_entropy_differences(token_entropies_df, post_df):
    difference_columns = defaultdict(lambda:[])

    risk_columns = [col for col in token_entropies_df.columns if 'no_risk' not in col and ('risk' in col or 'annotations' in col)]
    minuends = [c for c in ['no_risk_all', 'no_risk_sw'] if c in token_entropies_df.columns]
    
    for post_id, row in token_entropies_df.iterrows():   
        difference_columns['post_id'].append(post_id)
        tr_diffs = {}

        for col in risk_columns:
            for c in minuends:
                    
                if type(row[col]) == np.ndarray:
                    tr_diffs[c] = row[c] - row[col]                    
                else:
                    tr_diffs[c] = None

            colname = ''.join([i[0].upper() + i[1:] for i in col.split('_risk_')])
            for c in minuends:
                cname = ''.join([i[0].upper() + i[1:] for i in c.split('_risk_')])
                difference_columns[f"{cname}-{colname}"].append(tr_diffs[c])

    difference_columns = pd.DataFrame(difference_columns).set_index('post_id')

    post_ents_and_ent_diffs = pd.concat([post_df, token_entropies_df, difference_columns], axis=1)
    if ARGS.save_intermediate_values:
        outfile = os.path.join(ARGS.outdir, 'post_ents_and_ent_diffs.json')
        print(f'SAVING {outfile}')    
        post_ents_and_ent_diffs.to_json(outfile)


    return post_ents_and_ent_diffs
    
def index_post_tokens(post_ents_and_ent_diffs):
    test_on_test = defaultdict(lambda:[])
    sample2message_token_idx = []
    tdata_message = []
    for post_id, row in tqdm(post_ents_and_ent_diffs.iterrows(), total=len(post_ents_and_ent_diffs), desc="Mapping token idx to message idx"):
        test_on_test['post_id'].append(post_id)

        tokens = row.tokens
        # while '\"' in tokens:
        #     tokens.remove('\"')

        message = ""
        tidx2midx = {}
        for tidx, token in enumerate(tokens):
            # tidx = i + 1 # because skipping the bos token
            token = token.strip().replace('<|endoftext|>', '')#.replace("����", "😊😓")#.replace("\ufffd", "’") # hacky thing to deal with ' having resulted in \ufffd after tokenizing/writing to file
            tstart = len(message)
            message += token
            tend = len(message)
            tidx2midx[tidx] = (tstart, tend)

        sample2message_token_idx.append(tidx2midx)
        message = message.replace("\"", "")
        tdata_message.append(message)  

    test_on_test['tidx2midx'] = sample2message_token_idx
    test_on_test['message_strip_t'] = tdata_message
    test_on_test = pd.DataFrame(test_on_test).set_index('post_id')
    return test_on_test

def index_sentence_tokens(sentences_df):
    sample2message_token_idx = []
    tdata_message = []
    post_ids = []
    for post_id, frame in tqdm(sentences_df.groupby('post_id'), total=sentences_df.post_id.nunique(), desc="Mapping sentence tokens"):
        msg = ""
        duidx2midx = {}
        for _, row in frame.iterrows():
            # if math.isnan(row['message']):
            if type(row['sentence']) != str:
                print(row['sentence'])
                continue
                
            duid = row['message_id']
            du_msg = ''.join(row['sentence'].split())#.replace('’', '’’') # hacky thing to deal with ' having resulted in \ufffd after tokenizing/writing to file 
            dustart = len(msg)
            msg += du_msg
            duend = len(msg)
            duidx2midx[duid] = (dustart, duend)
        sample2message_token_idx.append(duidx2midx)
        # msg = msg.replace('&', '&amp;')
        msg = msg.replace('<', '&lt;')
        msg = msg.replace('’', '��')
        msg = msg.replace('“', '��')
        msg = msg.replace('”', '��')
        msg = msg.replace("\"", "")
        msg = msg.replace('\\t', '')
        msg = msg.replace("😊😓", '����')
        if msg[-3:] == 'nan':
            msg = msg[:-3]
        tdata_message.append(msg)
        post_ids.append(post_id)
    post_dunits = pd.DataFrame({'post_id':post_ids, 'duidx2midx':sample2message_token_idx, 'message_strip_du':tdata_message}).set_index('post_id')
    
    return post_dunits


def verify_maps(post_token_index_df, matching):
    print("If there are issues with matching the strings, the differences will print below, and there might be issues later...")
    cases = post_token_index_df[~matching][['message_strip_t', 'message_strip_du']].values
    for a,b in cases:     
        print('{}\n{}'.format(a,b))  
        for i,s in enumerate(difflib.ndiff(a, b)):
            if s[0]==' ': continue
            elif s[0]=='-':
                print(u'\tDelete "{}" from position {}'.format(s[-1],i))
            elif s[0]=='+':
                print(u'\tAdd "{}" to position {}'.format(s[-1],i))    
        print()  
    print("String difference checks complete.")

def map_tokens_to_sentences(post_token_index_df, post_ents_and_ent_diffs, sentences_df):
    sentence_tokens_df = defaultdict(lambda:[])
    ent_columns = post_ents_and_ent_diffs.columns[10:]


    for post_id, row in tqdm(post_token_index_df.iterrows(), total = len(post_token_index_df)):
        duidx2midx = row['duidx2midx']
        tidx2midx = row['tidx2midx']

        for duid, (dustart, duend) in duidx2midx.items():
            entropy_token_idx = []
            for tidx, (tstart, tend) in tidx2midx.items():
                # does tend actually matter? probably not when the messages are equal
                if tstart >= dustart and tstart < duend and tidx < len(row['tokens']):
                    entropy_token_idx.append(tidx)
                

            du_message = sentences_df[sentences_df['message_id'] == duid]['sentence'].values[0]
            du_tokens = [row['tokens'][tidx] for tidx in entropy_token_idx]
            du_token_ids = [row['token_ids'][tidx] for tidx in entropy_token_idx]

            sentence_tokens_df['message_id'].append(duid)
            sentence_tokens_df['post_id'].append(post_id)
            sentence_tokens_df['label'].append(row.label)
            sentence_tokens_df['sentence'].append(du_message)
            sentence_tokens_df['tokens'].append(du_tokens)
            sentence_tokens_df['token_ids'].append(du_token_ids)
            sentence_tokens_df['entropy_token_idx'].append(entropy_token_idx)

            for col in ent_columns:
                ent_list = post_ents_and_ent_diffs.loc[post_id, col]
                sentence_entropies = [ent_list[tidx] for tidx in entropy_token_idx]
                sentence_tokens_df[col].append(sentence_entropies)

    sentence_tokens_df = pd.DataFrame(sentence_tokens_df).set_index('message_id')
    if ARGS.save_intermediate_values:
        outfile = os.path.join(ARGS.outdir, f'{ARGS.test_group}_tokens_to_sentences.json')
        print(f'SAVING {outfile}')    
        sentence_tokens_df.to_json(outfile)
    return sentence_tokens_df


def aggregate(sentence_tokens_df):
    diff_columns = [col for col in sentence_tokens_df.columns if "-" in col]
    diff_aggregations = defaultdict(lambda:[])

    # get max, median, mean of all the differences
    for message_id, row in sentence_tokens_df.iterrows():
        diff_aggregations[f'message_id'].append(message_id)

        for col in diff_columns:
            diff_aggregations[f'{col}_mean'].append(np.mean(row[col]))
            diff_aggregations[f'{col}_median'].append(np.median(row[col]))
            diff_aggregations[f'{col}_max'].append(max(row[col]) if len(row[col]) > 0 else 0)

    diff_aggregations = pd.DataFrame(diff_aggregations).set_index('message_id')
    df = pd.concat([sentence_tokens_df, diff_aggregations], axis=1)
    return df, diff_aggregations
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog="python sentence_entropy.py",
                    description="Aggregates the token entropies at sentence level. Output file by default is {ARGS.test_group}_sentence_entropies.json")
    parser.add_argument('-dir', '--token_entropies_dir', default=os.path.join(MODELDIR, '_token_entropies'), type=str, help="Directory where you have the token entropies saved.")

    
    parser.add_argument('-o', '--outdir', type=str, help="Directory where you want to save the sentence entropies.", default=os.path.join(MODELDIR, '_sentence_entropies'))
    parser.add_argument('--all_messages', default=os.path.join(DATADIR, "clp24_all_messages.csv"), type=str, help="Path to test data file.")    
    parser.add_argument('-st', '--sent_tokenized_file', default=os.path.join(DATADIR, "clp24_SW_messages_sent_tokenized.csv"), type=str, help="Path to the data broken into sentences.")
    parser.add_argument('-te', '--test_group', choices=['any_risk_sw', 'annotations'], default='any_risk_sw', help="This specifies if we're using the internal annotations or the task submission data data.")
    parser.add_argument('-s', '--save_intermediate_values', action='store_true', help="Pass in if you want intermediate values to be written to a file.")



    ARGS = parser.parse_args()

    if ARGS.test_group == 'annotations':
        print("Pipeline not fully tested for internal annotations set yet.")
        POST_IDS = ['29d4au', '1ulqbn', '1y8cw4', 'ml312', '3iu651', '1end6q',
                    '34fj7p', '1811it', 'dtyiy', '3hmbre', '3i5n1a', '3j48ec',
                    '1kmzqg', 'rsk6g', '2hurih', '2nr23j', '1yc6u7', '1zif0m',
                    '2f9g97', '2nsqqm', '2swqdo', '3ghxts', '2dms1j', '2nzfj4',
                    '1vzleb', '3g31xw', '2deltm', '1el5om', '1jpm72', '2i87k0',
                    '2lsj9g', '2lxop0', '3hmw9z', '3iaxao', '27j41i', '37js2i',
                    '1j7nhr', '3eaouc', '2r2hkk', '35wrom', '2vhag4', '276h4l',
                    '3e1zc6', '2vxpos', '1wyqem', '1izo4t', '1d2gxo', '3ixiyn',
                    '17j9yv', 'd82jc'] # post ids that we have annotated internally
    elif ARGS.test_group == 'any_risk_sw':
        POST_IDS = ['2v6c9q', '3e8si7', '2j44ow', '2xe7hp', '3bhds2', '3d2ttx', '1kpxiu', '1d7nab', '1unus3', '3h9o7o', '3hh29z', 'vkxfb', '2nledl', '2dv5mw', '3ejsiv', '2il6xf', '3iagg6', '3h0z05', '2ff664', '32i9nz', '2a7wms', 'g7f7g', '280apu', '20kbpj', '2xa7jl', '17r00i', '3fuxue', '3c83gc', '3gcaik', '1azxoz', '39akaq', '22l8w4', '2mb3ku', '2h4b4q', '2xrnry', '2m9gm4', '2sq2me', '3ic4t8', '2vcol2', '32b7zk', '33kh5m', '35ylua', 'yaqxi', '2b3y50', '2booi6', '16d5de', '3fvr4l', '1shwnv', '31dz04', 'l5jxp', '3dv1gk', '2rp5bb', '319ioh', '2uquru', '1631gw', '30xti7', '32jo2k', '1tplbc', 'k76ov', '21mm33', '241gpm', '2v0w14', 'xfyos', '2f7wj4', '1ftvgt', '2j2t7i', '2zqq2c', '32o53l', '1wj80m', '3gizbj', '3gx2sj', '36eh73', '3am642', '2e5zho', '1w6g6u', '3hy1hk', '3i9p8b', '16xkll', '1ph5yt', '1gceob', '1t7se3', '3dthkh', '1y298o', '2jo2q0', '23tvqs', '2zienb', '2aqb2a', '2kvmof', 'mxci8', '13c9hl', '1455fd', '2u5z30', '36xc3s', '383n1f', '3cq8kt', '38bc2o', '26nqsc', '1getay', '1gwlbb', '2evvoo', '2uafbk', '2c7y41', '3gjkm6', '2iz5y7', '2h1vke', '35miyb', '2y7a6g', '3ci2ni', '2f27hy', '33r8b6', '1otrx6', '2802g5', '2z2q1f', '35lq71', '12umtk', '24u2gv', '1mjszi', '2d3de4', '3152g6', '3f9nd4', '3gk55c', '1pl51l', '2qlffz', '30reha', '1xpagm', '2017wz', '357o7f', '2scmvs', '2u91iq', '1zhg6p', '224i1n', '227n97', '2fwbqi', '25rnwu', '2x08xn', '3hxe6j', '2jfo9n', '2onplv', '278mzu', '1k3agb', '2yek5b', '2ng6fr', '1wb4qr', '2wsqbn', '2nmehc', '20d48v', '10llu1', '3896jf', '37jcax', '2dq4gx', '2tfdb4', '3ff64x', 'pnkh4', '15j1ft', '19jxq1', 'thn6n', 'ev6tp', '2gt6wk', '2hnma9', '3akx6o', '3f7iqw', '189w0w']
    
    os.makedirs(ARGS.outdir, exist_ok=True)

    main()
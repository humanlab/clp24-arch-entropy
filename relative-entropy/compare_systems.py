""" code by Allie Lahnala, please reach of if you have questions or suggestions: alahnala@gmail.com """
import argparse
import json
from tabulate import tabulate
from collections import defaultdict
import pandas as pd
import os
from config import DATADIR, MODELDIR



def main(sys1_submission, sys2_submission, maxcolwidths=70):
    s1_c2u = defaultdict(lambda:[])
    s2_c2u= defaultdict(lambda:[])

    for user_id, user_posts in user2posts.items():
        risk_level = USER_TABLE.loc[int(user_id), 'label']

        sys1_user_posts = sys1_submission[user_id]      
        sys2_user_posts = sys2_submission[user_id]        

        print(f"======= Evidence for user {user_id} | {risk_level} risk =======")
        for post_id in user_posts:
            sys1_highlights = sys1_user_posts[post_id]
            sys2_highlights = sys2_user_posts[post_id]

            # print("sys1:",  sys1_highlights)
            # print("sys2:", sys2_highlights)

            sys1_only = set(sys1_highlights) - set(sys2_highlights)
            sys2_only = set(sys2_highlights) - set(sys1_highlights)

            both = set(sys1_highlights).intersection(set(sys2_highlights))

            print(f"* post_id: {post_id} | risk_level: {risk_level}")
            
            user_prediction_table = defaultdict(lambda:[])

            sys1_highlights = list(sys1_only)
            sys2_highlights = list(sys2_only)
            print(f"\t* sys1 total: {len(sys1_highlights)} spans")
            print(f"\t* sys2 total: {len(sys2_highlights)} spans")

            max_len = max([len(sys1_highlights), len(sys2_highlights)])

            for i in range(max_len):        
                if i < len(sys1_highlights):
                    user_prediction_table['a#'].append(i+1)
                    user_prediction_table['sys1 only'].append(sys1_highlights[i])
                else:
                    user_prediction_table['a#'].append("")
                    user_prediction_table['sys1 only'].append("")

                if i < len(sys2_highlights):
                    user_prediction_table['b#'].append(i+1)
                    user_prediction_table['sys2 only'].append(sys2_highlights[i])
                else:
                    user_prediction_table['b#'].append("")
                    user_prediction_table['sys2 only'].append("")
            
            if "a#" in user_prediction_table:
                print(tabulate(pd.DataFrame(user_prediction_table).set_index('a#'), headers=user_prediction_table.keys(), maxcolwidths=maxcolwidths))
                print("\n")
        
            if len(both) > 0:
                both = {'Detected by both systems':list(both)}
                print(tabulate(pd.DataFrame(both), headers=both.keys(), maxcolwidths=2*maxcolwidths))
                print('\n')
        
        print("\n")
        s1_c2u[sum([len(sys1_user_posts[p]) for p in user_posts])].append(user_id)
        s2_c2u[sum([len(sys2_user_posts[p]) for p in user_posts])].append(user_id)
        

    if 0 in s1_c2u:
        print("WARNING: System 1 has users with no spans! Users:", s1_c2u[0])
    else:
        print("System 1 has spans for all users.")
    if 0 in s2_c2u:
        print("WARNING: System 2 has users with no spans! Users:", s2_c2u[0])
    else:
        print("System 2 has spans for all users.")
    


def reformat_obj(spans_per_user):
    d = {}
    for user_id in spans_per_user:
        d[user_id] = {}
        for post_object in spans_per_user[user_id]['posts']:
            d[user_id][post_object['post_id']] = post_object['highlights']
    return d



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog="Compare systems",
                    description="Compares the evidence spans extracted by each system")
    parser.add_argument('-a', '--sys1', help="path to system 1's submission file", required=True)
    parser.add_argument('-b', '--sys2', help="path to system 2's submission file", required=True)
    parser.add_argument('--submission_file_path', default=os.path.join(MODELDIR, '_submission_files'))
    parser.add_argument('-cw', '--maxcolwidth', help="determine the size of the highlight columns", default=70)
    parser.add_argument('-u', '--user_table_path', help="path to user table", default=os.path.join(DATADIR, "clp24_user_table.csv"))
    parser.add_argument('-te', '--test_group', choices=['any_risk_sw', 'annotations'], default='any_risk_sw', help="This specifies if we're using the internal annotations or the task submission data data.")

    


    ARGS = parser.parse_args()

    if ARGS.test_group == 'annotations':
        print('WARNING: Pipeline not fully tested on internal annotations.')
        template_path = os.path.join(DATADIR, "annotations_template.json")
    else:
        template_path = os.path.join(DATADIR, "submission_template.json")
    with open(template_path, 'r') as f:
        TEMPLATE = json.load(f)
    user2posts = reformat_obj(TEMPLATE)


    """ check paths to system output files """
    if not os.path.isfile(ARGS.sys1):
        wo_json = ARGS.sys1.replace('.json', '')
        ARGS.sys1 = os.path.join(ARGS.submission_file_path, f"{wo_json}.json")
        if not os.path.isfile(ARGS.sys1):
            print("ERROR: Can't find sys1 file.")
            quit()
    if not os.path.isfile(ARGS.sys2):
        wo_json = ARGS.sys2.replace('.json', '')
        ARGS.sys2 = os.path.join(ARGS.submission_file_path, f"{wo_json}.json")
        if not os.path.isfile(ARGS.sys2):
            print("ERROR: Can't find sys2 file.")
            quit()

    with open(ARGS.sys1, 'r') as f:
        sys1_user_spans = json.load(f)
    
    with open(ARGS.sys2, 'r') as f:
        sys2_user_spans = json.load(f)

    sys1_user_spans = reformat_obj(sys1_user_spans)
    sys2_user_spans = reformat_obj(sys2_user_spans)

    USER_TABLE = pd.read_csv(ARGS.user_table_path, index_col=0)


    main(sys1_user_spans, sys2_user_spans)
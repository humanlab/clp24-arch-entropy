""" code by Vasudha Varadarajan and Allie Lahnala, please reach of if you have questions or suggestions: alahnala@gmail.com """
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
import argparse
import pandas as pd
import os
import numpy as np
import json
from tabulate import tabulate
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from copy import deepcopy
from config import MODELDIR, DATADIR


def main():

    sent_df = pd.read_json(ARGS.sent_df_path)
    archetypes_sent = pd.read_csv(ARGS.archetypes_sent_file)
    archetypes_sent = archetypes_sent[archetypes_sent['message_id'].isin(sent_df.post_id.unique())]
    try:
        assert archetypes_sent.message_id.nunique() == sent_df.post_id.nunique()
    except:
        print("Sentence archetypes file is missing posts.")
        quit()

    
    """ 
    ===== map ids to join dataframes =====
    """
    archidx2sentidx = {}
    for post_id, frame in sent_df.groupby(by='post_id'):
        arch_frame = archetypes_sent[archetypes_sent['message_id'] == post_id]
        assert all(arch_frame.text.values == frame.sentence.values)
        for k,v in zip(arch_frame.index.values, frame.index.values):
            archidx2sentidx[k] = v

    archetypes_id_column = []
    for idx, _ in archetypes_sent.iterrows():
        new_id = archidx2sentidx[idx]
        archetypes_id_column.append(new_id)

    archetypes_sent['sent_id'] = archetypes_id_column
    archetypes_sent = archetypes_sent.set_index('sent_id')
    df = pd.concat([sent_df, archetypes_sent], axis=1)


    """ 
    ===== compute max archetype scores =====
    """
    df["max_arch_score"] = df.apply(lambda x: max([x[i] for i in archetypes]), axis=1)
    df["mean_arch_score"] = df.apply(lambda x: sum([x[i] for i in archetypes])/len(archetypes), axis=1)
    df["numeric_label"] = df["label"].apply(lambda x: {"Low": 1, "Moderate": 2, "High": 3}[x])

    # print_head(df, columns=['post_id', 'user_id', 'label', 'sentence', 'max_arch_score', 'mean_arch_score']) 

    """ 
    ===== Factor Analysis =====
    """

    print("====== Factor analysis on all archetypes ======")
    chi_square_value,p_value=calculate_bartlett_sphericity(df[archetypes])
    kmo_all,kmo_model=calculate_kmo(df[archetypes])

    print(f"\t* bartlett_sphericity(all archetypes)={chi_square_value}| pval={p_value}")
    print(f"\t* kmo(all archetypes)={kmo_model}")

    print("\n\t* factor analysis eigenvalues (all archetypes)")
    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df[archetypes])
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    for i in ev:
        print(f"\t\t{i}")

    print("\n====== Factor analysis on three joiner archetypes ======")
    print(f"\t* Archetypes:")
    for a in joiner_three_archs:
        print(f"\t\t* {a}")
    print_head(df, columns=joiner_three_archs) 

    fa = FactorAnalyzer(rotation=None)
    fa.fit(df[joiner_three_archs])
    # Check Eigenvalues
    print(f"\n\t* factor analysis eigenvalues ({joiner_three_archs})")
    ev, v = fa.get_eigenvalues()
    for i in ev:
        print(f"\t\t{i}")

    # fit a single factor
    j3_n_factors = 1
    fa = FactorAnalyzer(n_factors=j3_n_factors, rotation="varimax")
    fa.fit(df[joiner_three_archs])

    # shifting factor factor to use in products
    df["joiner3_arcs_factor"] =  fa.transform(df[joiner_three_archs]).mean(axis=1)
    if j3_n_factors == 1:
        df["joiner3_arcs_factor"] = df["joiner3_arcs_factor"] * -1
    df["joiner3_arcs_factor_shifted"] = abs(df["joiner3_arcs_factor"].min()) + df["joiner3_arcs_factor"]
    smooth = df["joiner3_arcs_factor_shifted"].sort_values().values[1] / 2
    df["joiner3_arcs_factor_shifted"] = smooth + df["joiner3_arcs_factor_shifted"]

    print(f"\n\t* top {ARGS.num_posts_to_show} posts with highest joiner3 factor")
    for i in df.sort_values(by="joiner3_arcs_factor", ascending=False).head(ARGS.num_posts_to_show).index:
        print(f"\t\t* {df.loc[i].sentence[:ARGS.max_output_len]}")

    print(f"\n\t* bottom {ARGS.num_posts_to_show} posts with lowest joiner3 factor")
    for i in df.sort_values(by="joiner3_arcs_factor", ascending=True).head(ARGS.num_posts_to_show).index:
        print(f"\t\t* {df.loc[i].sentence[:ARGS.max_output_len]}")



    print("\n====== Factor analysis on entropies ======")
    relevant_entropies = sorted([
        "NoSw-LowSw_max",
        "NoSw-HighSw_max",
        "NoSw-ModerateSw_max",
        "NoSw-AnySw_max"
    ])

    #rank of eigen matrix
    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df[relevant_entropies])
    # Check Eigenvalues
    print(f"\n\t* factor analysis eigenvalues ({relevant_entropies})")
    ev, v = fa.get_eigenvalues()
    for i in ev:
        print(f"\t\t{i}")

    # fit a single factor
    # fa = FactorAnalyzer(n_factors=1, rotation="varimax")
    ent_n_factors = 1
    fa = FactorAnalyzer(n_factors=ent_n_factors) # used 2 as last min choice because of issue with vals flipping
    fa.fit(df[relevant_entropies])
    #flatten the array
    df["entropy_factor"] =  fa.transform(df[relevant_entropies]).mean(axis=1)
    if ent_n_factors == 1:
        df["entropy_factor"] = df["entropy_factor"] * -1
    #print top 30 posts with highest entropy factor
    print(f"\n\t* top {ARGS.num_posts_to_show} posts with highest entropy factor")
    for i in df.sort_values(by="entropy_factor", ascending=False).head(ARGS.num_posts_to_show).index:
        print(f"\t\t* {df.loc[i].sentence[:ARGS.max_output_len]}")

    print(f"\n\t* bottom {ARGS.num_posts_to_show} posts with lowest entropy factor")
    for i in df.sort_values(by="entropy_factor", ascending=True).head(ARGS.num_posts_to_show).index:
        print(f"\t\t* {df.loc[i].sentence[:ARGS.max_output_len]}")


    # shifting entropy factor to use in products
    df["entropy_factor_shifted"] = abs(df["entropy_factor"].min()) + df["entropy_factor"]
    smooth = df["entropy_factor_shifted"].sort_values().values[1] / 2
    df["entropy_factor_shifted"] = smooth + df["entropy_factor_shifted"]




    """

    === Compute Policy Scores ===

    """

    # for products, use shifted values so there are no negatives
    df['prod_AC_ent'] = df['Acquired Capability - Ideation/Simulation'] * df['entropy_factor_shifted']
    df['prod_max_arc_ent'] = df['max_arch_score'] * df['entropy_factor_shifted']
    df['prod_j3_ent'] = df['joiner3_arcs_factor_shifted'] * df['entropy_factor_shifted']
    df['prod_j3_ent_ac'] = df['prod_j3_ent'] + df['Acquired Capability - Ideation/Simulation']
    df['prod_max_arc_ent_ac'] = df['prod_max_arc_ent'] + df['Acquired Capability - Ideation/Simulation']

    # for sums it doesn't matter if you use the shifted/original
    df['sum_AC_ent'] = df['Acquired Capability - Ideation/Simulation'] + df['entropy_factor']
    df['sum_max_arc_ent'] = df['max_arch_score'] + df['entropy_factor']
    df['sum_j3_ent'] = df['joiner3_arcs_factor'] + df['entropy_factor']


    """
    
    === Preview scores at threshold ===
    
    """
    df['sentence'] = df['sentence'].apply(lambda x: x.strip())
    print(df)
    print("\n====== Preview policy scores at threshold  ======")
    # for score in ['prod_AC_ent', 'prod_max_arc_ent', 'prod_j3_ent', 'sum_AC_ent', 'sum_max_arc_ent', 'sum_j3_ent']:
    # for score in ['prod_j3_ent+ac', 'prod_max_arc_ent+ac']:
    # for score in ['sum_max_arc_ent', 'prod_max_arc_ent', 'prod_AC_ent', 'prod_j3_ent+ac', 'prod_j3_ent']:
    for score in ['sum_max_arc_ent', 'prod_max_arc_ent', 'prod_AC_ent', 'prod_j3_ent_ac', 'prod_j3_ent', 'entropy_factor_shifted', 'max_arch_score', 'prod_max_arc_ent_ac']:
    # for score in ['entropy_factor_shifted']:
    # for score in ['max_arch_score', 'joiner3_arcs_factor', 'Acquired Capability - Ideation/Simulation']:
    # for score in ['prod_max_arc_ent', 'prod_j3_ent', 'prod_AC_ent', 'joiner3_arcs_factor']:
        print(f"\n* {score}")

        top_percentile = df[df[score] >= np.percentile(df[score], ARGS.percentile_threshold)]
        print("\t* Highest scoring sentences")
        ct = 1
        for i in top_percentile.sort_values(by=score, ascending=False).index[:ARGS.num_posts_to_show]:
            print(f"\t\t{ct}. {df.loc[i].sentence[:ARGS.max_output_len]}")
            ct+=1
        
        ct = 1
        print("\t* Lowest scoring sentences above threshold")
        for i in top_percentile.sort_values(by=score, ascending=True).index[:ARGS.num_posts_to_show]:
            print(f"\t\t{ct}. {df.loc[i].sentence[:ARGS.max_output_len]}")
            ct+=1

        ct = 1
        bottom_percentile = df[df[score] < np.percentile(df[score], ARGS.percentile_threshold)]
        print("\t* Highest scoring sentences below threshold (won't be highlighted)")
        for i in bottom_percentile.sort_values(by=score, ascending=False).index[:ARGS.num_posts_to_show]:
            print(f"\t\t{ct}. {df.loc[i].sentence[:ARGS.max_output_len]}")
            ct+=1

        
        

    """ Option to save dataframe with all these values """
    if ARGS.save_df:
        df.to_json(ARGS.save_df)    

    """
    
    === Write Submission Files ===
    
    """
    if ARGS.write:
        print("\n====== Writing submission files...  ======")
        for score in ['sum_max_arc_ent', 'prod_max_arc_ent', 'prod_AC_ent', 'prod_j3_ent_ac', 'prod_j3_ent', 'entropy_factor_shifted', 'max_arch_score', 'prod_max_arc_ent_ac']:

            top_percentile = df[df[score] >= np.percentile(df[score], ARGS.percentile_threshold)]
            write_submission(top_percentile, score, ARGS.percentile_threshold, df)
    
    return


def print_head(df, columns=None, rows=5, maxcolwidths=20, showindex=False):
    columns = columns if columns else list(df.columns[:3]) + list(df.columns[-3:])
    print(tabulate(df[columns].head(rows), maxcolwidths=maxcolwidths, headers=columns, showindex=showindex))




archetypes = [i.strip() for i in "Acquired Capability - Ideation/Simulation,Acquired Capability - Experiences of Endurance,Acquired Capability - Desensitization to Harm,Acquired Capability - High Tolerance for Physical Pain,Acquired Capability - Engagement in Risky Behaviors,Acquired Capability - Familiarity with Self-Harm Methods,Anger,Anhedonia,Availability of means,Aversive Self-awareness,Boredom,Circumstances Less than Expectations,Cognitive deconstruction,Embarassment,Escape,Internal Attributions,Loss/Failure,Medical Conditions,Mental Health,Nihilism,Omnibus Escape Theory,Perceived Burdensomness,Previous Attempts,Social Expectations / Sexuality,Substance Abuse,Unfairness,Thwarted Belongingness".split(",")]
joiner_three_archs = ['Perceived Burdensomness', 'Thwarted Belongingness', 'Acquired Capability - Ideation/Simulation']


def verify_spans_per_user(spans_per_user):
    for user_id in user2posts:
        assert user_id in spans_per_user
        for post in user2posts[user_id]:
            assert post in spans_per_user[user_id]
    print("Spans per user passes check")



def write_submission(top_percentile, score, p, df):
    spans_per_user = {}

    for user_id, user_posts in TEMPLATE.items():
        spans_per_user[user_id] = {}
        user_frame = df[df['user_id'] == int(user_id)]
        
        user_post_highlights = user_frame[user_frame.index.isin(top_percentile.index)]
        if len(user_post_highlights) == 0:
            spans_per_user[user_id] = {post_object['post_id']:[] for post_object in user_posts['posts']}
            item = user_frame.sort_values(by=score, ascending=False).iloc[0]
            post_id = item['post_id']
            sentence = item['sentence']
            spans_per_user[user_id][post_id].append(sentence)
        else:
            for post_object in user_posts['posts']:
                post_id = post_object['post_id']
                post_frame = df[df['post_id'] == post_id]
                post_highlights = post_frame[post_frame.index.isin(top_percentile.index)]
                spans_per_user[user_id][post_id] = list(post_highlights.sentence.values)

    verify_spans_per_user(spans_per_user)

    # fill template
    filled_template = deepcopy(TEMPLATE)
    for user_id, user_posts in filled_template.items():
        for post_object in user_posts['posts']:
            post_id = post_object['post_id']
            post_object['highlights'] = spans_per_user[user_id][post_id]

    outfile = os.path.join(ARGS.submission_file_path, f"{score}_{p}_submission.json")
    with open(outfile, "w") as f:
        #dump spans_per_user to json
        json.dump(filled_template, f, indent=4)
    
    print(f"wrote {outfile}")


def reformat_obj(spans_per_user):
    d = {}
    for user_id in spans_per_user:
        d[user_id] = {}
        for post_object in spans_per_user[user_id]['posts']:
            d[user_id][post_object['post_id']] = post_object['highlights']
    return d



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="python span_selection.py", description="Script to explore and decide on a span selection policy and select spans.")
    parser.add_argument('-s', '--save_df', default=None, type=str, help="Pass in a path to save the df with all the intermediate and final score values.")
    parser.add_argument('-arc', '--archetypes_sent_file', default=os.path.join(DATADIR, "archetypes_sent.csv"), type=str, help="Path to archetypes sentences file.")    
    parser.add_argument('-dir', '--sentence_entropy_dir', default=os.path.join(MODELDIR, '_sentence_entropies'), type=str, help="Path to directory that has the sentence entropy files.")
    parser.add_argument('-te', '--test_group', choices=['any_risk_sw', 'annotations'], default='any_risk_sw', help="This specifies if we're using the internal annotations or the task submission data data.")
    parser.add_argument('-df', '--sent_df_path', default=None, type=str, help="Path to sentence entropies file. This is an alternative to passing in --sentence_entropy_dir and --test_group.")
    parser.add_argument('-p', '--percentile_threshold', default=70, type=float)
    parser.add_argument('-o', '--submission_file_path', default=os.path.join(MODELDIR, '_submission_files'))
    parser.add_argument('-ml', '--max_output_len', default=100, type=int)
    parser.add_argument('--num_posts_to_show', default=5, type=int)
    parser.add_argument('--n_factors', default=1, type=int)
    parser.add_argument('-w', '--write', action='store_true', default=False)   

    ARGS = parser.parse_args()

    if ARGS.test_group == 'annotations':
        print('WARNING: Pipeline not fully tested on internal annotations.')
        template_path = os.path.join(DATADIR, "annotations_template.json")
    else:
        template_path = os.path.join(DATADIR, "submission_template.json")

    with open(template_path, 'r') as f:
        TEMPLATE = json.load(f)

    user2posts = reformat_obj(TEMPLATE)

    if not ARGS.sent_df_path:
        ARGS.sent_df_path = os.path.join(ARGS.sentence_entropy_dir, f'{ARGS.test_group}_sentence_entropies.json')
    
    if ARGS.write:
        os.makedirs(ARGS.submission_file_path, exist_ok=True)
    

    main()
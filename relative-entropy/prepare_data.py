""" 
* This script prepares the data for the language modeling scripts. 
* It will create two files:
    1) [DATADIR]/all_labels_messages_train.csv
    2) [DATADIR]/all_labels_messages_test.csv
* Change where to save the files in the two lines after the imports, if you like.
* Change the number of no risk posts to include in the test set, if you like. We go with a small sample.
"""
from config import DATADIR
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
import os
import random
random.seed(20)

# outfile paths
train_outfile = os.path.join(DATADIR, "all_labels_messages_train.csv")
test_outfile = os.path.join(DATADIR, "all_labels_messages_test.csv")

# how many no risk posts from r/SuicideWatch and other subreddits you want to include in your test set
no_risk_sw_test_size = 30 
no_risk_other_test_size = 1000

# Load the csv that contains all messages, including no risk.
all_messages = pd.read_csv(os.path.join(DATADIR, "clp24_all_messages.csv"))

# Load train and test post ids, and no risk post ids
test_post_ids = pd.read_csv(os.path.join(DATADIR, "clp24_all_messages_test.csv")).post_id.values
train_post_ids = pd.read_csv(os.path.join(DATADIR, "clp24_all_messages_train.csv")).post_id.values
no_risk_sw_post_ids = list(all_messages[all_messages['label'] == 'No'][all_messages['subreddit'] == 'SuicideWatch'].post_id.values)
no_risk_other_post_ids = list(all_messages[all_messages['label'] == 'No'][all_messages['subreddit'] != 'SuicideWatch'].post_id.values)
all_no_risk_post_ids = no_risk_sw_post_ids + no_risk_other_post_ids

# get a random sample of no risk posts for the train set
no_risk_test_pids = random.sample(no_risk_sw_post_ids, k=no_risk_sw_test_size) + random.sample(no_risk_other_post_ids, k=no_risk_other_test_size)
no_risk_train_pids = list(set(all_no_risk_post_ids) - set(no_risk_test_pids))

# Put no risk user posts together with the users with risk data
train_post_ids = list(train_post_ids) + list(no_risk_train_pids)
test_post_ids = list(test_post_ids) + list(no_risk_test_pids)
all_messages_train = all_messages[all_messages['post_id'].isin(train_post_ids)].sample(frac=1) # makes and shuffles the train data
all_messages_test = all_messages[all_messages['post_id'].isin(test_post_ids)].sample(frac=1) # makes and shuffles the test data

print('=== Train set posts per label ===\n', all_messages_train['label'].value_counts(), '\n')
print('=== Test set posts per label  ===\n', all_messages_test['label'].value_counts())

# save the data
all_messages_train.to_csv(train_outfile, index=False)
all_messages_test.to_csv(test_outfile, index=False)

print("\nCreated files:")
print(f"\t1) {train_outfile}")
print(f"\t2) {test_outfile}")
print('Data prep complete.')
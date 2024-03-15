# clp24-arch-entropy
Code used in our paper: Archetypes and Entropy: Theory-Driven Extraction of Evidence for Suicide Risk. Our submission for CLPsych 2024 Shared Task A and B.

## Archetypes

# Archetypes!

This is a library developed to run what might be called a "souped-up dictionary method" for psychological text analysis. Or any kind of text analysis, really.

The core idea behind *Archetypes* is that you pre-define a set of prototypical sentences that reflect the construct that you are looking to measure in a body of text. Using modern contextual embeddings, then, this library will aggregate your prototypes into an *archetypal* representation of your construct. Then, you can quantify texts in your corpus for their semantic similarity to your construct(s) of interest.

*Note*: For the curious: no, this approach not inspired by anything [Jungian](https://en.wikipedia.org/wiki/Jungian_archetypes) in nature. In the past, I've [said a few things](https://www.tandfonline.com/doi/full/10.1080/1047840X.2019.1633122?casa_token=cnHLr5uwiXUAAAAA:ACUCFK4tC9HXBBMlfhIFjfPxWLuCEK7owT3z_IBv2rZUa35fq4Z_rVkETqGO0wa1FqYjbmgxcRWxnw) about Jungian archetypes that have inspired scholars to write more than a few frustrated e-mails to me. Apologies to the Jungians.

# Installation

This package is easily installable via pip via the following command:

`pip install archetyper`


# Requirements
If you want to run the library without `pip` installing as shown above, you will need to first install the following packages:
- `numpy`
- `tqdm`
- `torch`
- `sentence_transformers`
- `nltk`

You can try to install these all in one go by running the following command from your terminal/cmd:

`pip install numpy tqdm torch sentence_transformers nltk`



# Examples

I have provided an example notebook in this repo that walks through the basic process of using this library, along with demonstrations of a few important "helper" functions to help you evaluate the statistical/psychometric qualities of your archetypes.




## Relative entropy 

All code is in the [relative-entropy](./relative-entropy/) directory. There is a script that will run the full pipeline: [run_entropy_pipeline.sh](./relative-entropy/run_entropy_pipeline.sh) (see steps 0 and 1 below before running it). The section below describes each step.


### Steps

0. Data: This pipeline assumes that the following files are in one directory:
    * clp24_all_messages.csv: This contains all messages from the dataset, including data from the *no risk* users. Prepare it to contain the `label` column with the post authors' risk levels and the `by` (which annotator group) column. For *no risk* users, set the label to "No".
    * clp24_all_messages_test.csv: The messages of the users designated for the test set for the shared task.
    * clp24_all_messages_train.csv: The messages of users not in the test set
    * clp24_SW_messages_sent_tokenized.csv: Sentences from each post in r/SuicideWatch (nltk.sent_tokenize applied to messages).

1. Create a file **config.py** with these contents:

    ```python
    '''config.py'''
    DATADIR = # string pointing to the path of the clpsych24 shared task data, e.g., "/data/clp24/" 
    MODELDIR = # string pointing to a path to a directory where you want to save the finetuning output, e.g., "/data/clp24/finetune-output/"
    ```

2. Prepare dataset for language modeling scripts
    ```bash
    python prepare_data.py
    ```
    * The script will create two files:
        1) DATADIR/all_labels_messages_train.csv
        2) DATADIR/all_labels_messages_test.csv
    * Change where to save the files in the two lines after the imports, if you like.
    * Change the number of no risk posts to include in the test set, if you like. We go with a small sample.

3. Finetune a language model on a group of CLPsych users based on risk level.

    * You are required to specify a group to finetune on. All options follow the format `{Risk level}_risk_{Subreddit set}`. *Risk level* can be no, low, moderate, high, or any (includes low, moderate, and high, but not no risk). *Subreddit set* can be sw (SuicideWatch posts only) or all (all posts by the users in the risk level group)

    
    * Example:
        ```bash
        # example
        python finetune.py --train_group high_risk_sw --model_output_dir data/model_output
        ```
    
    * Usage:
        ```
        Usage: 
        python finetune.py [-h] -g {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all} [-tr ALL_LABELS_MESSAGES_TRAIN] [-d MODEL_OUTPUT_DIR] [-k] [--logging_dir LOGGING_DIR] [--device DEVICE] [-m {distilgpt2}] [-e NUM_EPOCHS] [-lr LEARNING_RATE] [-wd WEIGHT_DECAY] [-ss SAVE_STRATEGY] [-es EVAL_STRATEGY] [--logging_steps LOGGING_STEPS] [--hidden_dropout_prob HIDDEN_DROPOUT_PROB] [--train_proportion TRAIN_PROPORTION]

        Options
            -h, --help
            -g, --train_group {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all}
            -tr ALL_LABELS_MESSAGES_TRAIN, --all_labels_messages_train ALL_LABELS_MESSAGES_TRAIN
                                    Path to training data file.
            -d, --model_output_dir MODEL_OUTPUT_DIR
                                    Directory for saving the model checkpoints. They will be saved at [args.dir]/[args.train_group]. Recommended to make this unique to your experiment.
            -k, --keep_annotations
                                    Pass in -k if you want to include the posts that we annotated internally.
            --logging_dir LOGGING_DIR
                                    Path to a logging dir if you don't want to use the default.
            --device DEVICE
            -m, --base_lm {distilgpt2}
                                    Name of pre-trained language model you want to finetune.
            -e, --num_epochs NUM_EPOCHS
            -lr LEARNING_RATE, --learning_rate LEARNING_RATE
            -wd, --weight_decay WEIGHT_DECAY
            -ss, --save_strategy SAVE_STRATEGY
            -es, --eval_strategy EVAL_STRATEGY
            --logging_steps LOGGING_STEPS
                                    Number of update steps between logs
            --hidden_dropout_prob HIDDEN_DROPOUT_PROB
                                    Dropout
            --train_proportion TRAIN_PROPORTION
                                    Proportion of data to use for train, the rest will be used for eval.
        ```

4. (Optional) Plot losses of the finetuned language models.
    ```bash
    python plot_loss.py 
    ```

    ```
    Options:
        -h, --help            show this help message and exit
        -lm LM_TO_PLOT [LM_TO_PLOT ...], --lm_to_plot LM_TO_PLOT [LM_TO_PLOT ...]
                                Name of the language models you want to plot (i.e. the 'train group'). If passing in nothing, it will do all found LMs in --model_dir path.
        -d, --model_dir MODEL_DIR 
                                Directory where the lms are. Defaults to MODELDIR specified in config.py.
        -o, --plot_path PLOT_PATH
                                Directory where you want to save the plot images. Defaults to MODELDIR/_loss_plots.
    ```


5. Compute token entropies using one group's language model on another group's test data.

    TODO: optimize implementation, current implementation is inefficient

    Example:

    ```bash
    # example

    # compute losses from each group model on their own test data
    python compute_entropy.py --train_group any_risk_sw --test_group any_risk_sw;

    # compute losses from each group model on any risk data
    python compute_entropy.py --train_group no_risk_sw --test_group any_risk_sw;
    ```

    ```
    Options:
        -h, --help            show this help message and exit
        -tr {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all}, --train_group {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all}
                                This specifies the language model that we'll use to compute losses.
        -te {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all,annotations}, --test_group {low_risk_sw,high_risk_sw,moderate_risk_sw,low_risk_all,high_risk_all,moderate_risk_all,no_risk_sw,no_risk_all,any_risk_sw,any_risk_all,annotations}
                                This specifies the data that the language model will be run on for losses.
        -d MODEL_OUTPUT_DIR, --model_output_dir MODEL_OUTPUT_DIR
                                Directory where the finetuned models are saved.
        -o TOKEN_ENTROPIES_DIR, --token_entropies_dir TOKEN_ENTROPIES_DIR
                                Directory path where you want to save the token entropies.
        --all_labels_messages_train ALL_LABELS_MESSAGES_TRAIN
                                Path to training data file.
        --all_labels_messages_test ALL_LABELS_MESSAGES_TEST
                                Path to test data file.
        -m {distilgpt2,lsanochkin/deberta-large-feedback,microsoft/deberta-base}, --base_lm {distilgpt2,lsanochkin/deberta-large-feedback,microsoft/deberta-base}
        -lm {lm,mlm}, --lm_type {lm,mlm}
                                lm: CausalLM | mlm: MaskedLM. Not implemented for mlm yet.
        --device DEVICE
        -sw SW_SIZE, --sw_size SW_SIZE
                                Context window size preceding target token.
        --checkpoint_selection CHECKPOINT_SELECTION
                                Strategy for choosing the model checkpoint to load. Use a) min_eval_loss if you want to choose based on the minimum loss on the val set during training, b) last to use the last checkpoint, or c) a path to
                                the checkpoint directory, e.g., model_dir/train_group/checkpoint-1000.
    ```

6. Map token entropies to sentences. Script aggregates the token entropies at sentence level. Output file by default is {ARGS.test_group}_sentence_entropies.json.

    ```bash
    python sentence_entropy.py
    ```

    ```
    Options:
        -h, --help            show this help message and exit
        -dir TOKEN_ENTROPIES_DIR, --token_entropies_dir TOKEN_ENTROPIES_DIR
                                Directory where you have the token entropies saved.
        -o ARGS.OUTDIR, --ARGS.outdir ARGS.OUTDIR
                                Directory where you want to save the sentence entropies.
        --all_messages ALL_MESSAGES
                                Path to test data file.
        -st SENT_TOKENIZED_FILE, --sent_tokenized_file SENT_TOKENIZED_FILE
                                Path to the data broken into sentences.
        -te {any_risk_sw,annotations}, --test_group {any_risk_sw,annotations}
                                This specifies the data that the language model will be run on for losses.
        -s, --save_intermediate_values
                                Pass in if you want intermediate values to be written to a file.
    ```

7. Explore span selection policies and write submission file with chosen policy. This step may involve your manual changes to the script to adjust your policies or create new ones, but code is written there that will output samples based on your policies.

    ```bash
    python span_selection.py # run with -w if you want to write submission files with the policies implemented in the script.
    ```

8. (optional) Use the compare_systems.py script to compare the highlighted evidence per user from two systems.

    ```bash
    # example
    python compare_systems.py -a max_arch_score_70_submission -b prod_max_arc_ent_ac_70_submission
    ```

    ```
    Options:
        -h, --help            show this help message and exit
        -s SAVE_DF, --save_df SAVE_DF
                                Pass in a path to save the df with all the intermediate and final score values.
        -arc ARCHETYPES_SENT_FILE, --archetypes_sent_file ARCHETYPES_SENT_FILE
                                Path to archetypes sentences file.
        -dir SENTENCE_ENTROPY_DIR, --sentence_entropy_dir SENTENCE_ENTROPY_DIR
                                Path to directory that has the sentence entropy files.
        -te {any_risk_sw,annotations}, --test_group {any_risk_sw,annotations}
                                This specifies if we're using the internal annotations or the task submission data data.
        -df SENT_DF_PATH, --sent_df_path SENT_DF_PATH
                                Path to sentence entropies file. This is an alternative to passing in --sentence_entropy_dir and --test_group.
        -p PERCENTILE_THRESHOLD, --percentile_threshold PERCENTILE_THRESHOLD
        -o SUBMISSION_FILE_PATH, --submission_file_path SUBMISSION_FILE_PATH
        -ml MAX_OUTPUT_LEN, --max_output_len MAX_OUTPUT_LEN
        --num_posts_to_show NUM_POSTS_TO_SHOW
        --n_factors N_FACTORS
        -w, --write
    ```

9. TODO: (optional) Make latex visual of highlighted spans


# Citation

This method is originally described in the following forthcoming paper:
```
@inproceedings{varadarajan_archetypes_2024,
	address = {St. Julians, Malta},
	title = {Archetypes and {Entropy}: {Theory}-{Driven} {Extraction} of {Evidence} for {Suicide} {Risk}},
	booktitle = {Proceedings of the {Tenth} {Workshop} on {Computational} {Linguistics} and {Clinical} {Psychology}},
	publisher = {Association for Computational Linguistics},
	author = {Varadarajan, Vasudha and Lahnala, Allison and Ganesan, Adithya V. and Dey, Gourab and Mangalik, Siddharth and Bucur, Ana-Maria and Soni, Nikita and Rao, Rajath and Lanning, Kevin and Vallejo, Isabella and Flek, Lucie and Schwartz, H. Andrew and Welch, Charles and Boyd, Ryan L.},
	year = {2024},
}
```

The citation above will be updated once the paper is actually published 😊
    

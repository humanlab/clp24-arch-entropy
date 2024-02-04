""" code by Allie Lahnala, please reach of if you have questions or suggestions: alahnala@gmail.com """
import os
from glob import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
from collections import defaultdict
from config import MODELDIR

def main():

    print("Checkpoint of minimum eval loss:")
    for lm in lms_to_plot:
        trainer_state_file, checkpoint_steps = get_checkpoint_states(model_dir, lm)

        if not trainer_state_file:
            continue

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

        path_to_min_loss_checkpoint = get_closest_to_min_eval_loss(df, checkpoint_steps, model_dir, lm)


        print(f"\t{lm}: step =", df.sort_values(by='Eval loss').index[0], " | epoch =", df.sort_values(by='Eval loss')['epoch'].values[0], " | path to closest checkpoint:", path_to_min_loss_checkpoint)

        xs = np.array(df.index)
        series1 = np.array(df['Eval loss'])
        s1mask = np.isfinite(series1)

        series2 = np.array(df['Train loss'])
        s2mask = np.isfinite(series2)

        plt.clf()
        plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='.', label='Eval loss')
        plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker = '.', label='Train loss')
        plt.legend(loc="lower left")
        plt.title(f"{lm} loss")
        plt.xlabel("Train step")
        plt.ylabel("Loss")

        save_path = os.path.join(plot_path, f"{lm}_eval.png")
        plt.savefig(save_path)



def get_checkpoint_states(lms_path, lm): 
    """
    Returns:
        - the path to the last trainer state (it includes all loss checkpoints)
        - list of checkpoint step values
    """
    checkpoint_path = os.path.join(lms_path, lm, "checkpoint-*", 'trainer_state.json') 
    checkpoint_dirs = glob(checkpoint_path)
    if len(checkpoint_dirs) == 0:
        print(checkpoint_path)
        print(f"No trainer state found for {lm}")
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
                    prog="plot_loss.py",
                    description="Plots train and eval losses for the language models you pass in.")
    parser.add_argument('-lm', '--lm_to_plot', help="Name of the language models you want to plot. If passing in nothing, it will do all found LMs in --model_dir path.", nargs='+', default=['*'], required=False)
    parser.add_argument('-d', '--model_dir', help="Directory where the lms are.", default=MODELDIR)
    parser.add_argument('-o', '--plot_path', help="Directory where you want to save the plot images.", default=os.path.join(MODELDIR, '_loss_plots'))
    

    ARGS = parser.parse_args()

    if ARGS.lm_to_plot[0] == '*':
        """do all that are in ARGS.model_dir"""
        trainer_state_paths = glob(os.path.join(ARGS.model_dir, '*', '*', 'trainer_state.json'))
        lms_to_plot = list(set([p.split('/')[-3] for p in trainer_state_paths]))
    else:
        lms_to_plot = [lm.replace(',', '') for lm in ARGS.lm_to_plot]
    

    model_dir = ARGS.model_dir[:-1] if ARGS.model_dir[-1] == '/' else ARGS.model_dir
    plot_path = ARGS.plot_path
    os.makedirs(plot_path, exist_ok=True)

    main()